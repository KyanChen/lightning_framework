custom_imports = dict(imports=['mmseg.datasets', 'mmdet.models'], allow_failed_imports=False)
# train max_num_instance=140
# test max_num_instance=96
sub_model = [
        'sam.mask_decoder.iou_token',
        'sam.mask_decoder.building_token',
        'sam.mask_decoder.iou_prediction_head',
        'sam.mask_decoder.building_probability_head',
    ]
max_epochs = 300

optimizer = dict(
    type='AdamW',
    sub_model=sub_model,
    lr=0.0001,
    weight_decay=1e-3
)

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
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    )
]

evaluator = dict(
    type='JaccardIndex',
    task='multiclass',
    num_classes=2,
    ignore_index=255,
    average='none'
)

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
num_classes = 150
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        deep_stem=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_cfg = dict(
    type='SegPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    sam='vit_h',
    sam_checkpoint='pretrain/sam/sam_vit_h_4b8939.pth',
    need_train_names=sub_model,
    loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
    loss_dice=dict(
        type='mmdet.DiceLoss',
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=1.0),
)

logger = dict(
    type='WandbLogger',
    project='building',
    group='test',
    name='E20230410_0'
)
# logger = None


callbacks = [
    dict(
        type='ModelCheckpoint',
        save_last=True,
        mode='max',
        monitor='metric_1',
        save_top_k=5,
        filename='epoch_{epoch}-iou_{metric_1:.4f}'
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
    default_root_dir='results/exp/E20230410_0',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=30,
    check_val_every_n_epoch=1,
    benchmark=True,
    # sync_batchnorm=True,

    # fast_dev_run=True,
    # limit_train_batches=1,
    # limit_val_batches=0,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=2,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=1,
    # gradient_clip_val=None,
    # gradient_clip_algorithm=None,
    # deterministic=None,
    # inference_mode: bool=True,
    # use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=False,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)

crop_size = (1024, 1024)
train_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations', reduce_zero_label=False, label_id_map={255: 1}),
    dict(
        type='mmseg.RandomResize',
        scale=(2048, 512),
        ratio_range=(1.0, 3.0),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=crop_size),
    dict(type='mmseg.Resize', scale=crop_size),
    dict(type='mmseg.RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='mmseg.PackSegInputs')
]

test_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='mmseg.Resize', scale=crop_size),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='mmseg.LoadAnnotations', reduce_zero_label=False, label_id_map={255: 1}),
    dict(type='mmseg.PackSegInputs')
]


train_batch_size_per_gpu = 1
train_num_workers = 2
test_batch_size_per_gpu = 1
test_num_workers = 2
persistent_workers = True


# data_parent = '/data1/kyanchen/datasets/'
data_parent = '/mnt/search01/dataset/cky_data/'
data_root = data_parent+'WHU/'
train_data_prefix = 'train/'
val_data_prefix = 'test/'

dataset_type = 'mmseg.ADE20KDataset'
metainfo = dict(classes=('building',), palette=[(0, 0, 255)])

val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(img_path=val_data_prefix+'image', seg_map_path=val_data_prefix+'label'),
            # indices=16,
            test_mode=True,
            pipeline=test_pipeline,
        )
    )

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(img_path=train_data_prefix+'image', seg_map_path=train_data_prefix+'label'),
            # indices=16,
            test_mode=False,
            pipeline=train_pipeline,
        )
    ),
    val_loader=val_loader,
    test_loader=val_loader
)

