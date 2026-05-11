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
