import copy
from typing import Dict, List, Optional
from torch import Tensor
from mmdet.models.builder import DETECTORS
from mmfewshot.detection.models import MetaRCNN
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as trans
from mmfewshot.detection.datasets.coco import COCO_SPLIT


@DETECTORS.register_module()
class FSAD(MetaRCNN):
    def __init__(self, *args, with_refine=False, **kwargs):
        super().__init__(*args, **kwargs)
        # refine results for COCO. We do not use it for VOC.
        self.with_refine = with_refine

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)

        # stop gradient at RPN
        query_feats_rpn = [x.detach() for x in query_feats]
        query_feats_rcnn = query_feats

        support_feats = self.extract_support_feat(support_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_rpn,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_rpn,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats_rcnn,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.model_init()

        query_feats = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals
        bbox_results = self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)
        if self.with_refine:
            return TestMixins.refine_test(bbox_results, img_metas)
        else:
            return bbox_results


class PCB:
    def __init__(self, class_names, model="RN101", templates="a photo of a {}"):
        super().__init__()
        self.device = torch.cuda.current_device()

        # image transforms
        self.expand_ratio = 0.1
        self.trans = trans.Compose([
            trans.Resize([224, 224], interpolation=3),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # CLIP configs
        import clip
        self.class_names = class_names
        self.clip, _ = clip.load(model, device=self.device)
        self.prompts = clip.tokenize([
            templates.format(cls_name)
            for cls_name in self.class_names
        ]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(self.prompts)
            self.text_features = F.normalize(text_features, dim=-1, p=2)

    def load_image_by_box(self, img_path, boxes):
        image = Image.open(img_path).convert("RGB")
        image_list = []
        for box in boxes:
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - w * self.expand_ratio)
            y1 = max(0, y1 - h * self.expand_ratio)
            x2 = x2 + w * self.expand_ratio
            y2 = y2 + h * self.expand_ratio
            sub_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            sub_image = self.trans(sub_image).to(self.device)
            image_list.append(sub_image)
        return torch.stack(image_list)

    @torch.no_grad()
    def __call__(self, img_path, boxes):
        images = self.load_image_by_box(img_path, boxes)

        image_features = self.clip.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t()
        return logits_per_image.softmax(dim=-1)


class TestMixins:
    def __init__(self):
        self.pcb = None

    def refine_test(self, results, img_metas):
        if not hasattr(self, 'pcb'):
            self.pcb = PCB(COCO_SPLIT['ALL_CLASSES'], model='ViT-B/32')
            # exclue ids for COCO
            self.exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                                46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

        boxes_list, scores_list, labels_list = [], [], []
        for cls_id, result in enumerate(results[0]):
            if len(result) == 0:
                continue
            boxes_list.append(result[:, :4])
            scores_list.append(result[:, 4])
            labels_list.append([cls_id] * len(result))

        if len(boxes_list) == 0:
            return results

        boxes_list = np.concatenate(boxes_list, axis=0)
        scores_list = np.concatenate(scores_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        logits = self.pcb(img_metas[0]['filename'], boxes_list)

        for i, prob in enumerate(logits):
            if labels_list[i] not in self.exclude_ids:
                scores_list[i] = scores_list[i] * 0.5 + logits[i, labels_list[i]] * 0.5

        j = 0
        for i in range(len(results[0])):
            num_dets = len(results[0][i])
            if num_dets == 0:
                continue
            for k in range(num_dets):
                results[0][i][k, 4] = scores_list[j]
                j += 1

        return results


