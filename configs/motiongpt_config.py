optimizer = dict(type='AdamW', lr=0.0001, weight_decay=1e-3)
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        by_epoch=True,
        begin=1,
        end=200,
    )
]

block_size = 64

model_cfg = dict(
    type='MotionGPTPLer',
    block_size=block_size,
    max_frames=64,
    mean_std_info=f'../data/lafan1_train_mean_std_info_{block_size}.pkl',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
    ),
    rotation_proj=dict(
        type='Sirens',
        in_channels=9,
        out_channels=768,
        base_channels=256,
        num_inner_layers=2,
    ),
    position_proj=dict(
        type='Sirens',
        in_channels=3,
        out_channels=768,
        base_channels=256,
        num_inner_layers=2,
    ),
    spatial_transformer=dict(
        type='TransformerEncoderNeck',
        model_dim=768,
        with_cls_token=True,
        num_encoder_layers=3
    ),
    temporal_transformer=dict(
        type='HFGPTTransformerDecoderNeck',
        model_name='distilgpt2',
        from_pretrained=False,
        update_kwargs=dict(
                max_position_embeddings=block_size,
                hidden_size=768,
            )
    ),
    head=dict(
        type='MotionGPTHead',
        in_channels=768,
        out_channels=dict(
            rot_6d=22*6*2,
            diff_root_zyx=3*2,
        ),
        # rotation_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        global_position_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        rotation_loss=dict(type='UncertaintyRegressionLoss', choice='smooth_l1', loss_weight=1.0),
        root_position_loss=dict(type='UncertaintyRegressionLoss', choice='smooth_l1', loss_weight=1.0)
    ),
)

logger = dict(
    type='WandbLogger',
    project='MotionGPT',
    group='test',
    name='E20230403_1'
)

trainer_cfg = dict(
    compiled_model=False,
    default_root_dir='results/exp',
    max_epochs=200,
    logger=logger,
    # strategy='ddp_find_unused_parameters_true',
    devices=1,
    # precision='32',
    # precision='16-mixed',
    log_every_n_steps=100,
    # limit_train_batches=1,
    # fast_dev_run=True,
    limit_val_batches=0,
    # gradient_clip_algorithm
    # gradient_clip_val
    # ckpt_path=None,
    # fast_dev_run=False,
    # limit_train_batches=0.1,
    # limit_val_batches=0.01,
    # enable_model_summary=False,
    # profiler="simple",
)

train_batch_size_per_gpu = 64
train_num_workers = 8
test_batch_size_per_gpu = 4
test_num_workers = 2
persistent_workers = False
datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type='BvhDataset',
            block_size=block_size,
            test_mode=False,
            data_root='../data/lafan1/',
        )
    ),
    predict_loader=dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type='BvhDataset',
            block_size=block_size,
            test_mode=True,
            data_root='../data/lafan1/',
            n_prompt_frames=10,
            n_offset=1000,
            # indices=8,
        )
    )
)

visualizer = dict(
    type='MotionVisualizer',
    save_dir='results/vis'
)







# class_name = ('ship', )
# num_classes = len(class_name)
# metainfo = dict(classes=class_name, palette=[(0, 0, 255)])
# env_cfg = dict(
#     cudnn_benchmark=True,
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl'),
# )
# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(type='WandbVisBackend',
#          init_kwargs=dict(
#              project='DyTiSDet',
#              group='YOLOv8',
#              name='E20230313_1'
#          )
#          )
# ]
# visualizer = dict(
#     type='mmdet.DetLocalVisualizer',
#     vis_backends=vis_backends,
#     name='visualizer')
# log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
# log_level = 'INFO'
# load_from = None
# resume = False
# file_client_args = dict(backend='disk')
#
# # data_parent = '/mnt/search01/dataset/cky_data'
# data_parent = '/expand_data/datasets'
# data_root = data_parent+'/Levir-ShipV2_Slices/train_test_slices_split/'
# train_data_prefix = 'train/'  # Prefix of train image path
# val_data_prefix = 'test/'  # Prefix of val image path
# # anno_path = '/mnt/search01/usr/chenkeyan/codes/dytisdet/DyTiSDet_yolo/data_infos/DyTiSDet/annotations'
# anno_path = '/data/kyanchen/dytisdet/DyTiSDet_yolo/data_infos/DyTiSDet/annotations'
# train_ann_file = anno_path+'/train.json'
# val_ann_file = anno_path+'/test.json'
#
# train_batch_size_per_gpu = 128
# train_num_workers = 4
# persistent_workers = True
#
# # -----train val related-----
# # Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
# base_lr = 0.05
# max_epochs = 1000  # Maximum training epochs
# # Disable mosaic augmentation for final 10 epochs (stage 2)
# close_mosaic_epochs = 10
#
# model_test_cfg = dict(
#     # The config of multi-label for multi-class prediction.
#     multi_label=True,
#     # The number of boxes before NMS
#     nms_pre=30000,
#     score_thr=0.001,  # Threshold to filter out boxes.
#     nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
#     max_per_img=300)  # Max number of detections of each image
#
# # ========================Possible modified parameters========================
# # -----data related-----
# img_scale = (512, 512)  # width, height
# # Dataset type, this will be used to define the dataset
# dataset_type = 'MNIST'
# # Batch size of a single GPU during validation
# val_batch_size_per_gpu = 128
# # Worker to pre-fetch data for each single GPU during validation
# val_num_workers = 4
#
# # Config of batch shapes. Only on val.
# # We tested YOLOv8-m will get 0.02 higher than not using it.
# batch_shapes_cfg = None
# # You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# # batch_shapes_cfg = dict(
# #     type='BatchShapePolicy',
# #     batch_size=val_batch_size_per_gpu,
# #     img_size=img_scale[0],
# #     # The image scale of padding should be divided by pad_size_divisor
# #     size_divisor=32,
# #     # Additional paddings for pixel scale
# #     extra_pad_ratio=0.5)
#
# # -----model related-----
# # The scaling factor that controls the depth of the network structure
# deepen_factor = 0.33
# # The scaling factor that controls the width of the network structure
# widen_factor = 0.5
# # Strides of multi-scale prior box
# strides = [8, 16, 32]
# # The output channel of the last stage
# last_stage_out_channels = 1024
# num_det_layers = 3  # The number of model output scales
# norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
#
# # -----train val related-----
# affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# # YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
# max_aspect_ratio = 100
# tal_topk = 10  # Number of bbox selected in each level
# tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
# tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# # TODO: Automatically scale loss_weight based on number of detection layers
# loss_cls_weight = 0.5
# loss_bbox_weight = 7.5
# # Since the dfloss is implemented differently in the official
# # and mmdet, we're going to divide loss_weight by 4.
# loss_dfl_weight = 1.5 / 4
# lr_factor = 0.01  # Learning rate scaling factor
# weight_decay = 0.0005
# # Save model checkpoint and validation intervals in stage 1
# save_epoch_intervals = 10
# # validation intervals in stage 2
# val_interval_stage2 = 1
# # The maximum checkpoints to keep.
# max_keep_ckpts = 5
#
#
# albu_train_transforms = [
#     dict(type='Blur', p=0.01),
#     dict(type='MedianBlur', p=0.01),
#     dict(type='ToGray', p=0.01),
#     dict(type='CLAHE', p=0.01)
# ]
#
# pre_transform = [
#     dict(type='LoadImageFromFile', file_client_args=file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True)
# ]
#
# last_transform = [
#     dict(
#         type='mmdet.Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         }),
#     dict(type='YOLOv5HSVRandomAug'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]
#
# train_pipeline = [
#     *pre_transform,
#     dict(
#         type='Mosaic',
#         img_scale=img_scale,
#         pad_val=114.0,
#         pre_transform=pre_transform),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
#         max_aspect_ratio=max_aspect_ratio,
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2),
#         border_val=(114, 114, 114)),
#     *last_transform
# ]
#
# train_pipeline_stage2 = [
#     *pre_transform,
#     dict(type='YOLOv5KeepRatioResize', scale=img_scale),
#     dict(
#         type='LetterResize',
#         scale=img_scale,
#         allow_scale_up=True,
#         pad_val=dict(img=114.0)),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
#         max_aspect_ratio=max_aspect_ratio,
#         border_val=(114, 114, 114)), *last_transform
# ]
#
# train_dataloader = dict(
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     # collate_fn=dict(type='yolov5_collate'),
#     dataset=dict(
#         type='MNIST',
#         data_prefix='data/mnist/', test_mode=False
#     )
# )
#
# test_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=file_client_args),
#     dict(type='YOLOv5KeepRatioResize', scale=img_scale),
#     dict(
#         type='LetterResize',
#         scale=img_scale,
#         allow_scale_up=False,
#         pad_val=dict(img=114)),
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'pad_param'))
# ]
#
# val_dataloader = dict(
#     batch_size=val_batch_size_per_gpu,
#     num_workers=val_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=metainfo,
#         test_mode=True,
#         data_prefix=dict(img=val_data_prefix),
#         ann_file=val_ann_file,
#         pipeline=test_pipeline,
#         batch_shapes_cfg=batch_shapes_cfg))
#
# test_dataloader = val_dataloader
# # test_dataloader = dict(
# #     batch_size=val_batch_size_per_gpu,
# #     num_workers=val_num_workers,
# #     persistent_workers=persistent_workers,
# #     pin_memory=True,
# #     drop_last=False,
# #     sampler=dict(type='DefaultSampler', shuffle=False),
# #     dataset=dict(
# #         type=dataset_type,
# #         data_root=data_root+'../application_slices/',
# #         metainfo=metainfo,
# #         test_mode=True,
# #         data_prefix=dict(img=''),
# #         ann_file=anno_path+'/app.json',
# #         pipeline=test_pipeline,
# #         batch_shapes_cfg=batch_shapes_cfg))
#
# param_scheduler = None
#
#
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=5),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='mmdet.DetVisualizationHook'),
#     param_scheduler=dict(
#         type='YOLOv5ParamSchedulerHook',
#         scheduler_type='linear',
#         lr_factor=lr_factor,
#         max_epochs=max_epochs),
#     checkpoint=dict(
#         type='CheckpointHook',
#         interval=save_epoch_intervals,
#         save_best='auto',
#         max_keep_ckpts=max_keep_ckpts)
# )
#
# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0001,
#         update_buffers=True,
#         strict_load=False,
#         priority=49),
#     dict(
#         type='mmdet.PipelineSwitchHook',
#         switch_epoch=max_epochs - close_mosaic_epochs,
#         switch_pipeline=train_pipeline_stage2)
# ]
#
# val_evaluator = dict(
#     type='mmdet.CocoMetric',
#     proposal_nums=(100, 1, 10),
#     ann_file=anno_path+'/test.json',
#     metric='bbox')
# test_evaluator = val_evaluator
#
# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=max_epochs,
#     val_interval=save_epoch_intervals,
#     dynamic_intervals=[((max_epochs - close_mosaic_epochs),
#                         val_interval_stage2)])
#
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

