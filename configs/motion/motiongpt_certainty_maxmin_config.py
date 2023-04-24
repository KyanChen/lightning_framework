optimizer = dict(type='AdamW', lr=0.0001, weight_decay=1e-3)

max_epochs = 150
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

block_size = 256
max_frames_predict = 128
model_cfg = dict(
    type='MotionGPTPLer',
    block_size=block_size,
    max_frames_predict=max_frames_predict,
    n_prompt_tokens=40,
    min_max_norm=True,
    mean_std_info=f'data/lafan1_train_mean_std_info_{block_size}.pkl',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
    ),
    # rotation_proj=dict(
    #     type='Sirens',
    #     in_channels=9,
    #     out_channels=768,
    #     base_channels=256,
    #     num_inner_layers=2,
    # ),
    # position_proj=dict(
    #     type='Sirens',
    #     in_channels=3,
    #     out_channels=768,
    #     base_channels=256,
    #     num_inner_layers=2,
    # ),
    rotation_proj=dict(
        type='LinearProj',
        in_channels=9,
        out_channels=768,
        base_channels=256,
        num_inner_layers=2,
    ),
    position_proj=dict(
        type='LinearProj',
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
            rot_6d=22*6,
            diff_root_zyx=3,
        ),
        loss='certainty_loss',
        return_certainty=True,
        uncertainty_beta=10,
        global_position_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        rotation_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        root_position_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        # out_channels=dict(
        #     rot_6d=22*6*2,
        #     diff_root_zyx=3*2,
        # ),
        # loss='uncertainty_loss',
        # global_position_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
        # rotation_loss=dict(type='UncertaintyRegressionLoss', choice='l2', loss_weight=1.0),
        # root_position_loss=dict(type='UncertaintyRegressionLoss', choice='l2', loss_weight=1.0)
    ),
)
exp_name = 'E20230419_0'
# logger = dict(
#     type='WandbLogger',
#     project='MotionGPT',
#     group='certain',
#     name=exp_name
# )
logger = None

callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/motiongpt/{exp_name}/checkpoints',
        monitor='val_loss',
        save_last=True,
        mode='min',
        save_top_k=5,
        filename='epoch_{epoch}-valloss_{val_loss:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    ),
    dict(
        type='MotionVisualizer',
        save_dir=f'results/vis_certainty_maxmin_{exp_name}',
        fps=29,
    )
]


trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=[0],
    default_root_dir='results/tmp',
    # default_root_dir=f'results/motiongpt/{exp_name}',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,

    # limit_val_batches=0,
    check_val_every_n_epoch=1,
    benchmark=True,
    sync_batchnorm=True,

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
train_batch_size_per_gpu = 32
train_num_workers = 8
test_batch_size_per_gpu = 32
test_num_workers = 8
persistent_workers = True
datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
        dataset=dict(
            type='BvhDataset',
            block_size=block_size,
            test_mode=False,
            data_root='data/lafan1/',
            n_offset=10,
        )
    ),
    val_loader=dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
        dataset=dict(
            type='BvhDataset',
            block_size=block_size,
            test_mode=False,
            phase='val',
            data_root='data/lafan1/',
            n_offset=50,
        )
    ),
    predict_loader=dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
        dataset=dict(
            type='BvhDataset',
            block_size=block_size,
            test_mode=True,
            phase='predict',
            data_root='data/lafan1/',
            n_offset=3000,
        )
    )
)

