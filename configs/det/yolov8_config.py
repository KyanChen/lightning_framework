custom_imports = dict(imports=['mmyolo.datasets'], allow_failed_imports=False)

base_lr = 0.05
lr_factor = 0.01
weight_decay = 0.0005
# optimizer = dict(type='AdamW', lr=0.0001, weight_decay=1e-3)
optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=weight_decay, nesterov=True)

max_epochs = 600
param_scheduler = None


last_stage_out_channels = 1024
deepen_factor = 0.33
widen_factor = 0.5
num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
num_classes = 1
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

strides = [8, 16, 32]
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4


evaluator_ = dict(
    type='MeanAveragePrecision',
    box_format='xyxy',
    iou_type="bbox",
    iou_thresholds=[0.5],
    rec_thresholds=[0.5],
    max_detection_thresholds=[1, 10, 100],
    average='none'
)

evaluator = dict(
    val_evaluator=evaluator_,
)

model_cfg = dict(
    type='YoloPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    data_preprocessor=dict(
        type='mmyolo.YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmyolo.YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    neck=dict(
        type='mmyolo.YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    head=dict(
        type='mmyolo.YOLOv8Head',
        head_module=dict(
            type='YOLOv8SIRENSHeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            target_size=(64, 64),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[512//64]*3),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[512//64]*3),
        bbox_coder=dict(type='mmyolo.DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmyolo.IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='mmyolo.BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=dict(
        # The config of multi-label for multi-class prediction.
        multi_label=True,
        # The number of boxes before NMS
        nms_pre=30000,
        score_thr=0.001,  # Threshold to filter out boxes.
        nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
        max_per_img=300
    )  # Max number of detections of each image
)

img_scale = (512, 512)
affine_scale = 0.5
max_aspect_ratio = 100
file_client_args = dict(backend='disk')
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='mmyolo.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmyolo.LoadAnnotations', with_bbox=True)
]
last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='mmyolo.YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='mmyolo.LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)),
    *last_transform
]

logger = dict(
    type='WandbLogger',
    project='DyTiSDet',
    group='yolov8_dynamic',
    name='E20230417_0'
)
# logger = None

close_mosaic_epochs = 40
callbacks = [
    dict(
        type='ModelCheckpoint',
        monitor='valmap_50_0',
        save_last=True,
        mode='max',
        save_top_k=6,
        filename='epoch_{epoch}-map50_{valmap_50_0:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    ),
    dict(
        type='PipelineSwitchHook',
        # switch_epoch=0,
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2
    ),
    dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    dict(
        type='EMAHook',
        ema_type='mmyolo.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False
    ),
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    strategy="auto",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=8,
    default_root_dir='results/DyTiSDet/E20230417_0',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,
    check_val_every_n_epoch=1,
    benchmark=True,
    sync_batchnorm=True,

    # fast_dev_run=True,
    # limit_train_batches=5,
    # limit_val_batches=5,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=2,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=1,
    gradient_clip_val=25,
    gradient_clip_algorithm='norm',
    # deterministic=None,
    # inference_mode: bool=True,
    use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=False,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)


train_pipeline = [
    *pre_transform,
    dict(
        type='mmyolo.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]


test_pipeline = [
    dict(type='mmyolo.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='mmyolo.LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='mmyolo.LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_batch_size_per_gpu = 80
train_num_workers = 4
test_batch_size_per_gpu = 80
test_num_workers = 4
persistent_workers = True

data_parent = '/mnt/search01/dataset/cky_data'
# data_parent = '/Users/kyanchen/datasets'
# data_parent = '/expand_data/datasets'
data_root = data_parent+'/Levir-ShipV2_Slices/train_test_slices_split/'
train_data_prefix = 'train/'  # Prefix of train image path
val_data_prefix = 'test/'  # Prefix of val image path
anno_path = '/mnt/search01/usr/chenkeyan/codes/dytisdet/DyTiSDet_yolo/data_infos/DyTiSDet/annotations'
# anno_path = '/Users/kyanchen/codes/dytisdet/DyTiSDet_yolo/data_infos/DyTiSDet/annotations'
# anno_path = '/data/kyanchen/dytisdet/DyTiSDet_yolo/data_infos/DyTiSDet/annotations'

train_ann_file = anno_path+'/train.json'
val_ann_file = anno_path+'/test.json'

dataset_type = 'mmyolo.YOLOv5CocoDataset'
metainfo = dict(classes=('ship', ), palette=[(0, 0, 255)])
val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            metainfo=metainfo,
            data_root=data_root,
            ann_file=val_ann_file,
            data_prefix=dict(img=val_data_prefix),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            # indices=8,
            pipeline=test_pipeline)
)

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        collate_fn=dict(type='yolov5_collate'),
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            metainfo=metainfo,
            data_root=data_root,
            ann_file=train_ann_file,
            data_prefix=dict(img=train_data_prefix),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            # indices=8,
            pipeline=train_pipeline)
    ),
    val_loader=val_loader,
)

