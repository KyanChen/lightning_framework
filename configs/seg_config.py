sub_model = [
        'sam.mask_decoder.iou_token',
        'sam.mask_decoder.building_token',
        'sam.mask_decoder.iou_prediction_head',
        'sam.mask_decoder.building_probability_head',
    ],

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
        T_max=100,
        by_epoch=True,
        begin=1,
        end=100,
    )
]

model_cfg = dict(
    type='SegPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
    ),
    sam='vit_h',
    sam_checkpoint='../pretrain/sam/sam_vit_h_4b8939.pth',
    need_train_names=sub_model,
)

# logger = dict(
#     type='WandbLogger',
#     project='MotionGPT',
#     group='test',
#     name='E20230403_1'
# )

logger = None
callbacks = [
    dict(
        type='ModelCheckpoint',
        monitor='val_loss',
        save_top_k=5,
    ),
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    strategy="auto",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=1,
    default_root_dir='results/exp',
    max_epochs=100,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=1,
    # check_val_every_n_epoch=1,
    benchmark=True,
    # sync_batchnorm=True,

    # fast_dev_run=True,
    limit_train_batches=1,
    limit_val_batches=0,
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
    dict(type='mmseg.LoadAnnotations', reduce_zero_label=True, label_id_map={255: 1}),
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
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]


train_batch_size_per_gpu = 2
train_num_workers = 0
test_batch_size_per_gpu = 4
test_num_workers = 0
persistent_workers = False


data_parent = '/data1/kyanchen/datasets/'
data_root = data_parent+'WHU/'
train_data_prefix = 'train/'
val_data_prefix = 'test/'

dataset_type = 'mmseg.ADE20KDataset'
metainfo = dict(classes=('building',), palette=[(0, 0, 255)])

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
            reduce_zero_label=True,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(img_path=train_data_prefix+'image', seg_map_path=train_data_prefix+'label'),
            indices=16,
            test_mode=False,
            pipeline=train_pipeline,
        )
    ),
)

