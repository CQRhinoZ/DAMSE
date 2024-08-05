_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_nwpuv2_400.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotNWPUV2DefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        num_novel_shots=3,
        num_base_shots=3,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=6000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=2000)
optimizer = dict(type='SGD', lr=0.001)
optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(
    warmup_iters=100, step=[
        6000,
    ])

runner = dict(max_iters=6000)
# runner = dict(max_iters=1)

#my
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    frozen_parameters=[
        'backbone', 'rpn'],

    neck=dict(
        type='MyFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
        # The type of RPN head is 'RPNHead'. For more details, please refer to
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12
        # type='my_RPNHead',
        type='RPNHead',
        # The input channels of each input feature map,
        # this is consistent with the output channels of neck
        in_channels=256,
        # Feature channels of convolutional layers in the head.
        feat_channels=256,
        anchor_generator=dict(  # The config of anchor generator
            # Most of methods use AnchorGenerator, For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10
            type='AnchorGenerator',
            # Basic scale of the anchor, the area of the anchor in one position
            # of a feature map will be scale * base_sizes
            scales=[8],
            # The ratio between height and width.
            ratios=[0.5, 1.0, 2.0], #长宽比
            # The strides of the anchor generator. This is consistent with the FPN
            # feature strides. The strides will be taken as base_sizes if base_sizes is not set.
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(  # Config of box coder to encode and decode the boxes during training and testing
            # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods. For more details refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9
            type='DeltaXYWHBBoxCoder',
            # The target means used to encode and decode boxes
            target_means=[0.0, 0.0, 0.0, 0.0],
            # The standard variance used to encode and decode boxes
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # Config of loss function for the classification branch
        loss_cls=dict(
            # Type of loss for classification branch.
            type='CrossEntropyLoss',
            # RPN usually perform two-class classification,
            # so it usually uses sigmoid function.
            use_sigmoid=True,
            # Loss weight of the classification branch.
            loss_weight=1.0),
        # Config of loss function for the regression branch.
        loss_bbox=dict(
            # Type of loss, we also support many IoU Losses and smooth L1-loss. For implementation refer to
            # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56
            type='L1Loss',
            # Loss weight of the regression branch.
            loss_weight=1.0)),


    roi_head=dict(
        bbox_head=dict(
            num_classes=10,  # Number of classes for classification
        )),


)



# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

load_from = ('/ai/zyr/mmfewshot-main/work_dirs/my_r101_fpn_nwpu-split1_base-training/'
             'base_model_random_init_bbox_head.pth')
