# dataset settings
dataset_type = 'CWCDataset'  
test_dataset_type = 'CustomDataset'  # otherwise reports registry error
data_root = 'data/cwc_v1'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='annotations',
        split='train.txt',
        ignore_index=0,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='annotations',
        split='val.txt',
        ignore_index=0,
        pipeline=test_pipeline),
    test=dict(
        test_mode=True, 
        type=test_dataset_type,
        data_root='/home/ubuntu/embedded_ads',
        img_dir='frame_over',  # 'embedded_ads_frames',
        # ann_dir='annotations',
        split='frame_over.txt', # test_batch1.txt',  # test_batch2.txt',
        img_suffix='.png',
        # seg_map_suffix='.png',
        reduce_zero_label=False,
        classes = ('background', 'ads'),
        palette = [[0,0,0], [255,255,255]],
        ignore_index=0,
        pipeline=test_pipeline))

