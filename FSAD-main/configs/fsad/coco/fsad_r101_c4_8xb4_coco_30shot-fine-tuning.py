_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../fsad_r101_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=30,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='30SHOT')],
            num_novel_shots=30,
            num_base_shots=30,
        )),
    model_init=dict(num_novel_shots=30, num_base_shots=30))
evaluation = dict(interval=20000)
checkpoint_config = dict(interval=20000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[20000])
runner = dict(max_iters=20000)
# load_from = 'path of base training model'
load_from = 'work_dirs/fsad_r101_c4_8xb4_coco_base-training/latest.pth'
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80, num_meta_classes=80)),
    with_refine=True,
    frozen_parameters=[
    'backbone', 'shared_head',  'aggregation_layer', 'rpn_head.rpn_conv',
])