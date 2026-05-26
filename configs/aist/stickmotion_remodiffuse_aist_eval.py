_base_ = ['./stickmotion_remodiffuse_aist.py']

data = dict(
    test=dict(
        test_mode=True,
        eval_cfg=dict(
            _delete_=True,
            shuffle_indexes=True,
            replication_times=2,
            replication_reduction='statistics',
            text_encoder_name='aist60',
            text_encoder_path='/mnt/data3_hdd/alex/FlowMimic/runs/aist_t2m_evaluator_deps/t2m/text_mot_match/model/finest.tar',
            motion_encoder_name='aist60',
            motion_encoder_path='/mnt/data3_hdd/alex/FlowMimic/runs/aist_t2m_evaluator_deps/t2m/text_mot_match/model/finest.tar',
            metrics=[
                dict(type='R Precision', batch_size=32, top_k=3),
                dict(type='Matching Score', batch_size=32),
                dict(type='FID'),
                dict(type='Diversity', num_samples=300),
                dict(type='MultiModality', num_samples=50, num_repeats=10, num_picks=5),
            ],
        ),
    ),
)
