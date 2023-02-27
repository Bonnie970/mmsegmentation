_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', '../_base_/datasets/cwc_v1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(pretrained='open-mmlab://contrib/mobilenet_v3_large')

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(interval=4000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)