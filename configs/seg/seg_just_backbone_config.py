custom_imports = dict(imports=['mmseg.datasets', 'mmdet.models'], allow_failed_imports=False)
# train max_num_instance=140
# test max_num_instance=96
sub_model = [
    # 'sam.mask_decoder.iou_token',
    # 'sam.mask_decoder.iou_prediction_head',
    # 'sam.mask_decoder.class_aware_token',
    # 'sam.mask_decoder.class_aware_head',
    'global_prompt',
    ]
max_epochs = 500

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

evaluator_ = dict(
    type='JaccardIndex',
    task='multiclass',
    num_classes=2,
    ignore_index=255,
    average='none'
)
evaluator = dict(
    train_evaluator=evaluator_,
    val_evaluator=evaluator_,
)

num_classes = 2
model_cfg = dict(
    type='SegPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    sam='vit_h',
    sam_checkpoint='pretrain/sam/sam_vit_h_4b8939.pth',
    points_per_side=None,
    only_img_encoder=True,
    global_prompt=True,
    need_train_names=sub_model,
    head=dict(
        type='BinarySemanticSegHead',
        # loss_mask=dict(
        #     type='mmdet.CrossEntropyLoss',
        #     use_sigmoid=True,
        #     reduction='mean',
        #     loss_weight=5.0),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=10.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
    )
)

logger = dict(
    type='WandbLogger',
    project='building',
    group='b_pred',
    name='E20230414_0'
)
# logger = False


callbacks = [
    dict(
        type='ModelCheckpoint',
        save_last=True,
        mode='max',
        monitor='valmulticlassjaccardindex_1',
        save_top_k=10,
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
    default_root_dir='results/building/E20230414_0',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,
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
    dict(type='mmseg.LoadAnnotations'),
    # dict(
    #     type='mmseg.RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(1.0, 3.0),
    #     keep_ratio=True),
    # dict(type='mmseg.RandomCrop', crop_size=crop_size),
    dict(type='mmseg.Resize', scale=crop_size),
    dict(type='mmseg.RandomFlip', prob=0.5),
    # dict(type='mmseg.PhotoMetricDistortion'),
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


train_batch_size_per_gpu = 6
train_num_workers = 2
test_batch_size_per_gpu = 6
test_num_workers = 2
persistent_workers = True


# data_parent = '/data1/kyanchen/datasets/'
# data_parent = '/Users/kyanchen/datasets/Building/'
data_parent = '/mnt/search01/dataset/cky_data/'
# data_parent = '../sample/'
data_root = data_parent+'WHU/'
train_data_prefix = 'train/'
val_data_prefix = 'test/'

dataset_type = 'BuildingExtractionDataset'
metainfo = dict(classes=('background_', 'building',), palette=[(0, 0, 0), (0, 0, 255)])

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

