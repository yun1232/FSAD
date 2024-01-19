_base_ = [
    './meta-rcnn_r50_c4.py',
]

custom_imports = dict(
    imports=[
        'model.detector',
        'model.roi_head',
        'model.bbox_head'],
    allow_failed_imports=False)

pretrained = 'data/pretrained/resnet101_caffe.pth'
# model settings
model = dict(
    type='FSAD',
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='FSADRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='FSADBBoxHead', num_classes=20, num_meta_classes=20)))
