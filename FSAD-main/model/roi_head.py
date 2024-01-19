from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead


class Generator(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.proc = nn.Linear(dim,dim)
        self.batch1 = nn.BatchNorm1d(dim)
        self.relu = nn.LeakyReLU()
        self.fc_mean = nn.Linear(dim, dim)
        self.fc_logit = nn.Linear(dim, dim)
        self.mid = nn.Linear(dim, dim)
        self.res = nn.Linear(dim, dim)
        self.batch2 = nn.BatchNorm1d(dim)
        self.sig = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, support: Tensor) -> List[Tensor]:

        x = self.proc(support)
        x = self.batch1(x)
        result = self.relu(x)
        means_feat = self.fc_mean(result)
        logit_feat = self.fc_logit(result)
        std = torch.exp(0.5 * logit_feat)
        eps = torch.randn_like(std)
        feat = eps * std + means_feat
        uncouple_info = std + means_feat
        distance = 1 + logit_feat - means_feat ** 2 - logit_feat.exp()
        y = self.mid(feat)
        y = self.res(y)
        y = self.batch2(y)
        new_feat = self.sig(y)

        return [new_feat, uncouple_info, distance]


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        validity = torch.sigmoid(self.fc2(z))
        return validity

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.res = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, query_x, support_x):
        B_x, C_x = query_x.shape
        B_y, C_y = support_x.shape
        query = query_x.view(query_x.shape[0], 1, query_x.shape[1])
        support = support_x.view(support_x.shape[0], 1, support_x.shape[1])

        assert B_y == 1
        q_x = self.q(query).reshape(B_x, 1, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        q_y = self.q(support).reshape(B_y, 1, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3)
        kv_x = self.kv(query).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
        kv_y = self.kv(support).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)

        k_x, v_x = kv_x[0], kv_x[1]
        k_y, v_y = kv_y[0], kv_y[1]
        k_y_ext = k_y.repeat(B_x, 1, 1, 1)
        v_y_ext = v_y.repeat(B_x, 1, 1, 1)
        k_cat_x = torch.cat((k_x, k_y_ext), dim=2)
        v_cat_x = torch.cat((v_x, v_y_ext), dim=2)

        attn_x_tmp = (q_x @ k_cat_x.transpose(-2, -1)) * self.scale
        attn_x = attn_x_tmp.softmax(dim=-1)
        #query = (attn_x @ v_cat_x).transpose(1, 2).reshape(B_x, C_x)
        #query = self.res(query)

        mask_x = torch.sum(attn_x_tmp, 3)
        mask_x = mask_x.view(attn_x.shape[0], self.num_heads)
        mask_x = mask_x.mean(1, True)

        k_x_avg = k_x.mean(0, True)
        v_x_avg = v_x.mean(0, True)
        k_cat_y = torch.cat((k_x_avg, k_y), dim=2)
        v_cat_y = torch.cat((v_x_avg, v_y), dim=2)

        attn_y_tmp = (q_y @ k_cat_y.transpose(-2, -1)) * self.scale
        attn_y = attn_y_tmp.softmax(dim=-1)
        support = (attn_y @ v_cat_y).transpose(1, 2).reshape(B_y, C_y)
        support = self.res(support)

        mask_y = torch.sum(attn_y_tmp, 3)
        mask_y = mask_y.view(attn_y.shape[0], self.num_heads)
        mask_y = mask_y.mean(1, True)

        mask = nn.Sigmoid().cuda()(mask_x)

        return query_x, support, mask

@HEADS.register_module()
class FSADRoIHead(MetaRCNNRoIHead):

    def __init__(self, dim=2048, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

        self.gen = Generator(dim)
        self.dis = Discriminator(dim)
        self.crossattn = CrossAttention(dim)

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]#[15,2048],1
        support_feat_rec, support_feat_inv, distance = self.gen(
            support_feat)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size#128
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            # class agnostic aggregation
            # random_index = np.random.choice(
            #     range(query_gt_labels[img_id].size(0)))
            # random_query_label = query_gt_labels[img_id][random_index]
            random_index = np.random.choice(
                range(len(support_gt_labels)))
            random_query_label = support_gt_labels[random_index]
            for i in range(support_feat.size(0)):
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat_inv[i].sigmoid().unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            # input support feature classification
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_rec)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        recons_loss = F.mse_loss(support_feat_rec, support_feat)
        validity = self.dis(support_feat_rec)
        adversarial_loss = -torch.log(validity + 1e-8).mean()
        generator_loss = -torch.log(1-(validity + 1e-8)).mean()
        gan_loss = adversarial_loss + generator_loss
        gan_weight = 0.025
        dis_weight = 0.00025
        dis_loss = torch.mean(-0.5 * torch.sum(distance, dim=1), dim=0)
        loss_enhance = {'loss_gan': recons_loss + gan_weight * gan_loss + dis_weight * dis_loss}
        loss_bbox.update(loss_enhance)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (128, 2048).
            support_roi_feats (Tensor): Support features with shape (1, 2048).
            roi_feats:(128, 2048, 1, 1)

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        new_roi_query, new_roi_support, mask=self.crossattn(query_roi_feats,support_roi_feats)
        support_roi_feat = 0.0001 * new_roi_support + support_roi_feats
        query_roi_feats = 0.0001 * new_roi_query * mask + query_roi_feats
       
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feat.view(1, -1, 1, 1))[0]
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1), query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            support_feat_rec, support_feat_inv, distance = self.gen(
                support_feat)
            bbox_results = self._bbox_forward(
                query_roi_feats, support_feat_inv.sigmoid())
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            # the official code use the first class background score as final
            # background score, while this code use average of all classes'
            # background scores instead.
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
