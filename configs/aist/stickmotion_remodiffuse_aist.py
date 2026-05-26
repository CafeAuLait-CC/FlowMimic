dataset_name = 'human_ml3d'
data_prefix = '/mnt/data3_hdd/alex/FlowMimic/prepared/aist_stickmotion'
point_len = 64
feat_dim = 256
norm_pose_dim = 22 * 3

stick_set = dict(
    train=dict(
        batch_size=256,
        epochs=15,
        lr=1e-4,
        workers=8,
        dataset_name=dataset_name,
    ),
    stickman_encoder=dict(
        point_len=point_len,
        in_dim=point_len * 2,
        out_dim=feat_dim,
        d_model=512,
        dropout=0.1,
        activation='relu',
        nhead=16,
        num_layers=5,
        ff_dim=1024,
    ),
    stickman_decoder=dict(
        in_dim=feat_dim,
        out_dim=norm_pose_dim,
        fcn_dims=[512, 512, 512, 512, 512],
        dropout=0.1,
        candidate_num=4,
    ),
    motion_encoder=dict(
        in_dim=263,
        out_dim=feat_dim,
        fcn_dims=[512, 512, 512, 512, 512],
        dropout=0.1,
    ),
    loss=dict(loss1=1., loss2=1., loss3=1.),
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=600)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

input_feats = 263
locus_dim = 10
max_seq_len = 196
latent_dim = 512
time_embed_dim = 2048
text_latent_dim = 256
ff_size = 1024
num_heads = 8
dropout = 0.
index_num = 3
crop_size = 196

data_keys = [
    'motion',
    'motion_mask',
    'motion_length',
    'sample_idx',
    'text_idx',
    'stickman_tracks',
    'locus',
    'stick_mask',
]
meta_keys = ['text', 'token']
train_pipeline = [
    dict(type='Crop', crop_size=crop_size),
    dict(type='StickThing', crop_size=crop_size),
    dict(
        type='Normalize',
        mean_path='/mnt/data3_hdd/alex/FlowMimic/prepared/aist_stickmotion/datasets/human_ml3d/mean.npy',
        std_path='/mnt/data3_hdd/alex/FlowMimic/prepared/aist_stickmotion/datasets/human_ml3d/std.npy',
    ),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys),
]

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='Stickmant2mDataset',
            dataset_name=dataset_name,
            data_prefix=data_prefix,
            pipeline=train_pipeline,
            ann_file='train.txt',
            motion_dir='motions',
            text_dir='texts',
            token_dir='tokens',
            clip_feat_dir=None,
            crop_size=crop_size,
        ),
        times=20,
    ),
    test=dict(
        type='Stickmant2mDataset',
        dataset_name=dataset_name,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        ann_file='test.txt',
        motion_dir='motions',
        text_dir='texts',
        token_dir='tokens',
        clip_feat_dir=None,
        crop_size=crop_size,
        eval_cfg=None,
        test_mode=False,
    ),
)

model = dict(
    type='MotionDiffusion',
    loss_weight=dict(stickman_w=1.0, locus_w=1.0),
    guidance=dict(repeat=3, layer_num=2, scale=1),
    index_num=index_num,
    motion_crop=[4, 4 + 21 * 9],
    model=dict(
        type='ReMoDiffuseTransformer',
        input_feats=input_feats,
        max_seq_len=max_seq_len,
        latent_dim=latent_dim,
        time_embed_dim=time_embed_dim,
        num_layers=6,
        condition_cfg=dict(text_p=0.7, stick_p=0.7, index_train_p=0.7),
        index_num=index_num,
        ca_block_cfg=dict(
            type='SemanticsModulatedAttention',
            latent_dim=latent_dim,
            text_latent_dim=text_latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            locus_dim=locus_dim,
            time_embed_dim=time_embed_dim,
            stick_latent_dim=latent_dim,
        ),
        ffn_cfg=dict(
            latent_dim=latent_dim,
            ffn_dim=ff_size,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        ),
        text_encoder=dict(
            pretrained_model='clip',
            latent_dim=text_latent_dim,
            num_layers=2,
            ff_size=2048,
            dropout=dropout,
            use_text_proj=False,
        ),
        multistick_encoder=dict(
            stick_encoder=stick_set['stickman_encoder'],
            weight='/mnt/data3_hdd/alex/FlowMimic/stickmotion/stickman/weight/kit_ml/split_weight/stickman_encoder.ckpt',
            d_model=feat_dim,
            out_dim=latent_dim,
        ),
        locus_encoder=dict(input_dim=4, latent_dim=locus_dim),
        scale_func_cfg=dict(
            coarse_scale=2.0,
            both_coef=0.52351,
            text_coef=-0.28419,
            retr_coef=2.39872,
        ),
    ),
    loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
    diffusion_train=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_large',
    ),
    diffusion_test=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_large',
        respace='15,15,8,6,6',
    ),
    inference_type='ddim',
)
