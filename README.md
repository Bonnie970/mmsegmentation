# Hotstar Cricket Embedded Ads Segmentation 
1. [https://hotstar.atlassian.net/browse/RES-2326](https://hotstar.atlassian.net/browse/RES-2326)
2. [https://hotstar.atlassian.net/browse/RES-2478](https://hotstar.atlassian.net/browse/RES-2478)
3. modify dataset config mmsegmentation/configs/_base_/datasets/cwc_v1.py based on mmsegmentation/mmseg/datasets/custom.py
4. modify overall config mmsegmentation/configs/mobilenet_v3/cwc_v1.py
5. when to eval, save checkpoint

## Train 
```bash 
conda activate mmdet
CONFIG_FILE=configs/mobilenet_v3/cwc_v1.py
bash tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir cwc_test_0
```

## Test 
run test

    - create test_over.txt `ls -1 ~/embedded_ads/frame_over | sed -e 's/\.png$//' > ~/embedded_ads/frame_over/frame_over.txt`
    
    - split into 1000 line chunks to avoid dataloader error (still don’t know why, always fails around 1.6k). `split -l 1000 frame_over.txt frame_over`
```bash 
conda activate mmdet
CONFIG_FILE=configs/mobilenet_v3/cwc_v1.py
CHECKPOINT_FILE=cwc_test_0/iter_40000.pth
for split in $(ls ~/embedded_ads/frame_over/frame_over_a*); do
	python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir cwc_test_over/mask --opacity 1 --gpu-id 0 \
	--cfg-options data.test.split=${split} data.test.data_root=/home/ubuntu/embedded_ads data.test.img_dir=frame_over
done
```