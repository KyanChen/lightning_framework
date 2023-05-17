custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models'], allow_failed_imports=False)

sub_model_train = [
    'neck',
    'decode_head',
    'data_preprocessor'
]

sub_model_optim = {
    'neck': {'lr_mult': 1},
    'decode_head': {'lr_mult': 1},
}

max_epochs = 300

optimizer = dict(
    type='AdamW',
    sub_model=sub_model_optim,
    lr=0.0002,
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

param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)

num_classes = 1
evaluator_ = dict(
    type='JaccardIndex',
    task='multiclass',
    num_classes=num_classes+1,
    ignore_index=255,
    average='none'
)

evaluator = dict(
    # train_evaluator=evaluator_,
    val_evaluator=evaluator_,
)


image_size = (1024, 1024)

data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=image_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)


model_cfg = dict(
    type='SemSegSAMPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    need_train_names=sub_model_train,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # type='vit_h',
        # checkpoint='pretrain/sam/sam_vit_h_4b8939.pth',
        type='vit_b',
        delete_submodel=['prompt_encoder', 'mask_decoder'],
        checkpoint='pretrain/sam/sam_vit_b_01ec64.pth',
    ),
    neck=dict(
        type='SAMAdaptor',
        in_channels=3,
        inner_dim=128,
        embed_dim=768,
        depth=12,
        # embed_dim=768,
        # depth=12,
        out_channels=256,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    decode_head=dict(
        type='SamSemSegHead',
        in_channels=256,
        num_classes=num_classes,
        ignore_index=255,
        threshold=0.5,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
)


task_name = 'whu_sem'
exp_name = 'E20230517_0'
logger = dict(
    type='WandbLogger',
    project=task_name,
    group='sam',
    name=exp_name
)
logger = None


callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/{task_name}/{exp_name}/checkpoints',
        save_last=True,
        mode='max',
        monitor='valmulticlassjaccardindex_0',
        save_top_k=2,
        filename='epoch_{epoch}-map_{valmulticlassjaccardindex_0:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
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
    devices=[7],
    default_root_dir=f'results/{task_name}/{exp_name}',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,
    check_val_every_n_epoch=20,
    benchmark=True,
    # sync_batchnorm=True,
    # fast_dev_run=True,

    # limit_train_batches=1,
    # limit_val_batches=0,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    num_sanity_val_steps=0,
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
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.RandomFlip', prob=0.5),
    # dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='mmseg.Resize', scale=image_size),
    dict(type='mmseg.PackSegInputs')
]

test_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='mmseg.Resize', scale=image_size),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.PackSegInputs')
]


train_batch_size_per_gpu = 2
train_num_workers = 0
test_batch_size_per_gpu = 2
test_num_workers = 0
persistent_workers = False

# data_root = '/Users/kyanchen/datasets/Building/WHU/'
data_root = '/mnt/search01/dataset/cky_data/WHU/'
train_data_prefix = 'train/'
val_data_prefix = 'test/'

dataset_type = 'BuildingExtractionDataset'

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
        data_root=data_root,
        data_prefix=dict(img_path=val_data_prefix + 'image', seg_map_path=val_data_prefix + 'label'),
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
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            data_root=data_root,
            data_prefix=dict(img_path=train_data_prefix + 'image', seg_map_path=train_data_prefix + 'label'),
            # indices=16,
            test_mode=False,
            pipeline=train_pipeline,
        )
    ),
    val_loader=val_loader,
    # test_loader=val_loader
    # predict_loader=val_loader
)