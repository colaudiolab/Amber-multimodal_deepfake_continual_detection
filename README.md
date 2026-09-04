# Amber-multimodal_deepfake_continual_detection
The official code of paper "Adaptive Affinity Memorization with Layer Mutation for Multimodal Deepfake Continual Detection"

## Training and evaluation
```bash
python main.py Amber --dataset MDCDDataset --base-ratio 0.25 --phases 4 --CL-type TIL\
    --csv-dir /Task_incremental_DB/4-cross_dataset_1_9 \
    --batch-size 16 --num-workers 4 --backbone GAT_video_audio \
    --memory-per-domain 500 --base-epochs 5 --learning-rate 0.001  \
    --IL-batch-size 16 --gpus 0
```
```bash
python main.py Amber --dataset MDCDDataset --base-ratio 0.25 --phases 4 --CL-type TIL\
    --csv-dir /Task_incremental_DB/4-cross_dataset_1_9 \
    --batch-size 16 --num-workers 4 --backbone LipSyncModel \
    --memory-per-domain 500 --base-epochs 10 --learning-rate 0.001  \
    --IL-batch-size 16 --gpus 0
```
```bash
python main.py Amber --dataset MDCDDataset --base-ratio 0.25 --phases 4 --CL-type TIL\
    --csv-dir /Task_incremental_DB/4-cross_dataset_1_9 \
    --batch-size 16 --num-workers 4 --backbone LTI \
    --memory-per-domain 500 --base-epochs 50 --learning-rate 0.001  \
    --IL-batch-size 16 --gpus 0
```

## Citation
```
@ARTICLE{Xiao2026Adaptive,
  author={Xiao, Man and Ye, Jianbin and Liu, Bo and Gao, Zijian and Chen, Wuyang and Li, Tao and Wang, Huaimin and Xu, Kele},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Adaptive Affinity Memorization With Layer Mutation for Multimodal Deepfake Continual Detection}, 
  year={2026},
  volume={21},
  number={},
  pages={6052-6067},
  keywords={Deepfakes;Signal detection;Modeling;Memory;Continuing education;Visualization;Training;Videos;Learning (artificial intelligence);Technology;Multimodal deepfake detection;continual learning;cross-modal gap;feature drift},
  doi={10.1109/TIFS.2026.3707760}}
```
```
# GB/T 7714
Xiao M, Ye J, Liu B, Gao Z, Chen W, Li T, Wang H, Xu K. Adaptive Affinity Memorization With Layer Mutation for Multimodal Deepfake Continual Detection[J/OL]. IEEE Transactions on Information Forensics and Security, 2026, 21: 6052-6067. DOI:10.1109/TIFS.2026.3707760.
```
