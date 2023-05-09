optimizer = dict(type='AdamW', lr=0.0002, betas=(0.5, 0.9), weight_decay=1e-3)

max_epochs = 200
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

nb_joints = 21
nb_code = 512
block_size = 128
pad_token = nb_code

model_cfg = dict(
    type='MotionLMGPTPLer',
    data_preprocessor=dict(
        type='BatchFixedSizePadTokenMaskGPT',
        pad_token=pad_token,
        p_token_keep=0.8,
        nb_code=nb_code,
    ),
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
    ),
    backbone=dict(
        type='MotionTransformer',
        num_vq=512,
        embed_dim=512,  # 263
        block_size=block_size,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4
    ),
    head=dict(
        type='MotionGPTPseudoHead',
        losses=dict(
            cls_loss=dict(type='mmdet.CrossEntropyLoss', ignore_index=pad_token),
        ),
    ),
)

task_name = 'motiongpt'
exp_name = 'E20230509_0'

logger = dict(
    type='WandbLogger',
    project=task_name,
    group='motionlmgpt',
    name=exp_name
)
# logger = None

callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        monitor='val_loss',
        save_last=True,
        mode='min',
        save_top_k=2,
        filename='epoch_{epoch}-valloss_{val_loss:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    ),
    dict(
        type='MotionKITVisualizer',
        num_joints=nb_joints,
        save_dir=f'results/{task_name}/{exp_name}/vis',
        cache_dir=f'cache_data/{task_name}/kit',
        fps=12,
    )
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    # strategy="auto",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=4,
    # default_root_dir='results/tmp',
    default_root_dir=f'results/{task_name}/{exp_name}',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,

    # limit_val_batches=0,
    check_val_every_n_epoch=1,
    benchmark=True,
    # sync_batchnorm=True,

    # fast_dev_run=True,
    # limit_train_batches=1,
    # limit_val_batches=1,
    # limit_test_batches=None,
    # limit_predict_batches=20,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=0,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=1,
    # gradient_clip_val=None,
    # gradient_clip_algorithm=None,
    # deterministic=None,
    # inference_mode: bool=True,
    use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=True,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)

# train_batch_size_per_gpu = 32
train_batch_size_per_gpu = 128
train_num_workers = 8
test_batch_size_per_gpu = 128
test_num_workers = 8
persistent_workers = True

# data_root = '/Users/kyanchen/codes/motion/KIT-ML'
data_root = '/mnt/search01/dataset/cky_data/KIT-ML'

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type='MotionGPTDataset',
            data_root=data_root,
            ann_file='train.txt',
            token_dir=f'cache_data/{task_name}/kit',
            block_size=block_size,
            n_offset=1,
            test_mode=False,
        )
    ),
    val_loader=dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type='MotionGPTDataset',
            data_root=data_root,
            ann_file='test.txt',
            token_dir=f'cache_data/{task_name}/kit',
            # dataset_name='kit',
            block_size=block_size,
            n_offset=5,
            test_mode=True,
        )
    ),
)

