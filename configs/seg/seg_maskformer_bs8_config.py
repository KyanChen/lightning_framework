custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models'], allow_failed_imports=False)

max_epochs = 300

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=1e-4
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
param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)

evaluator_ = dict(
    type='JaccardIndex',
    task='multiclass',
    num_classes=2,
    ignore_index=255,
    average='none'
)
evaluator = dict(
    # train_evaluator=evaluator_,
    val_evaluator=evaluator_,
)

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 1
data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255)
model = dict(
    type='mmseg.EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmseg.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        contract_dilation=True,
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    decode_head=dict(
        type='mmseg.MaskFormerHead',
        threshold=0.5,
        in_channels=[256, 512, 1024,
                     2048],  # input channels of pixel_decoder modules
        feat_channels=256,
        in_index=[0, 1, 2, 3],
        num_classes=num_classes,
        out_channels=256,
        num_queries=100,
        pixel_decoder=dict(
            type='mmdet.PixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # DetrTransformerDecoder
            return_intermediate=True,
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.1,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=1.0),
                    dict(
                        type='mmdet.FocalLossCost',
                        weight=20.0,
                        binary_input=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=1.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

model_cfg = dict(
    type='MMSegPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    whole_model=model,
)

exp_name = 'E20230428_0'
# logger = dict(
#     type='WandbLogger',
#     project='building',
#     group='maskformer',
#     name=exp_name
# )

logger = None


callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/building/{exp_name}/checkpoints',
        save_last=True,
        mode='max',
        monitor='valmulticlassjaccardindex_1',
        save_top_k=5,
        filename='epoch_{epoch}-iou_{valmulticlassjaccardindex_1:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    ),
    dict(
        type='SegVisualizer',
        save_dir=f'results/building/{exp_name}/visualizations',
    )
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="cpu",
    strategy="auto",
    # strategy="ddp",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=1,
    default_root_dir=f'results/building/{exp_name}',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=5,
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


train_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    # dict(
    #     type='mmseg.RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(1.0, 3.0),
    #     keep_ratio=True),
    # dict(type='mmseg.RandomCrop', crop_size=crop_size),
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
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.PackSegInputs')
]


train_batch_size_per_gpu = 8
train_num_workers = 8
test_batch_size_per_gpu = 8
test_num_workers = 8
persistent_workers = True


# data_parent = '/data1/kyanchen/datasets/'
data_parent = '/Users/kyanchen/datasets/Building/'
# data_parent = '/mnt/search01/dataset/cky_data/'
data_parent = 'samples/seg/'
data_root = data_parent+'WHU/'
train_data_prefix = 'train/'
val_data_prefix = 'test/'

dataset_type = 'BuildingExtractionDataset'
metainfo = dict(classes=('building', ), palette=[(0, 0, 255)])

load_sam_cache_from = 'cache_data/sam_data'
val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            phrase='val',
            # load_sam_cache_from=load_sam_cache_from,
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
            phrase='train',
            # load_sam_cache_from=load_sam_cache_from,
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
    # test_loader=val_loader
    predict_loader=val_loader
)

