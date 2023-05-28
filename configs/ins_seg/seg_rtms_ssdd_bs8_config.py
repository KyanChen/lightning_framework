custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models', 'mmdet.models'], allow_failed_imports=False)
# train max 71, min 1
# val max 56, min 1
max_epochs = 500

optimizer = dict(
    type='AdamW',
    lr=0.0005,
    weight_decay=1e-4
)

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=5e-4,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    )
    # dict(
    #     type='MultiStepLR',
    #     begin=1,
    #     end=max_epochs,
    #     by_epoch=True,
    #     milestones=[max_epochs//2, max_epochs*3//4],
    #     gamma=0.2)
]

param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)

# evaluator_ = dict(
#         type='MeanAveragePrecision',
#         iou_type='segm',
# )

evaluator_ = dict(
        type='CocoPLMetric',
        metric=['bbox', 'segm'],
        proposal_nums=[1, 10, 100]
)

evaluator = dict(
    # train_evaluator=evaluator_,
    val_evaluator=evaluator_,
    test_evaluator=evaluator_
)


image_size = (512, 512)
data_preprocessor = dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        mask_pad_value=0,
        pad_size_divisor=32
)

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

# model settings
model = dict(
    type='mmdet.RTMDet',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
    ),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        expand_ratio=0.5,
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='mmdet.RTMDetInsSepBNHead',
        num_classes=num_classes,
        in_channels=128,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=128,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='mmdet.DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='mmdet.DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5),
)

model_cfg = dict(
    type='MMDetPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    whole_model=model,
)

task_name = 'ssdd_ins'
exp_name = 'E20230528_2'
logger = dict(
    type='WandbLogger',
    project=task_name,
    group='rtms',
    name=exp_name
)
# logger = None


callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/{task_name}/{exp_name}/checkpoints',
        save_last=True,
        mode='max',
        monitor='valsegm_map_0',
        save_top_k=2,
        filename='epoch_{epoch}-map_{valsegm_map_0:.4f}'
        # mode='min',
        # monitor='train_loss',
        # save_top_k=2,
        # filename='epoch_{epoch}-trainloss_{train_loss:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    )
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    strategy="auto",
    # strategy="ddp",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=[0, 1, 2, 3],
    default_root_dir=f'results/{task_name}/{exp_name}',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=10,
    check_val_every_n_epoch=5,
    benchmark=True,
    # sync_batchnorm=True,
    # fast_dev_run=True,

    # limit_train_batches=1,
    # limit_val_batches=0,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=0,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=32,
    # gradient_clip_val=15,
    # gradient_clip_algorithm='norm',
    # deterministic=None,
    # inference_mode: bool=True,
    use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=False,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)


backend_args = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=image_size),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=image_size),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_batch_size_per_gpu = 8
train_num_workers = 4
test_batch_size_per_gpu = 8
test_num_workers = 4
persistent_workers = True

# # data_parent = '/Users/kyanchen/datasets/seg/VHR-10_dataset_coco/NWPUVHR-10_dataset/'
# data_parent = '/mnt/search01/dataset/cky_data/WHU'
# train_data_prefix = 'train/'
# val_data_prefix = 'test/'
# dataset_type = 'WHUInsSegDataset'

data_parent = '/mnt/search01/dataset/cky_data/SSDD'
dataset_type = 'SSDDInsSegDataset'

val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            # ann_file='annotations/WHU_building_test.json',
            # data_prefix=dict(img_path=val_data_prefix + '/image', seg_path=val_data_prefix + '/label'),
            ann_file='annotations/SSDD_instances_val.json',
            data_prefix=dict(img_path='imgs'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=test_pipeline,
            backend_args=backend_args))

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            # ann_file='annotations/WHU_building_train.json',
            # data_prefix=dict(img_path=train_data_prefix + '/image', seg_path=train_data_prefix + '/label'),
            ann_file='annotations/SSDD_instances_train.json',
            data_prefix=dict(img_path='imgs'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)
    ),
    val_loader=val_loader,
    test_loader=val_loader
    # predict_loader=val_loader
)