#!/bin/bash
# Out-of-fold segmenters, then predictions for every VQA image (runs/seg_oof/preds) and the pooled mIoU.
#   FOLDS="0 1 2 3 4" GPUS="1 3 5 7 6" ./run_seg_folds.sh        (one GPU per fold, same order)
# 2080 Tis (1,3,5,7): fp16, batch 2 x accum 2; 3090/A6000 (0,2,6): bf16, batch 4. GPU 4 faulted on 2026-10-13.
# The all-data model (--fold -1, runs/seg_all) is trained separately on the 3090.
# The cache is read from local RAID (NAS reads were the bottleneck); rebuild it with `seg_pipeline.py prep` and copy.
cd "$(dirname "$0")"
CACHE=${CACHE:-/media/data_dump/Anwar/btp_vqa_cache/seg_cache_official}
FOLDS=(${FOLDS:-0 1 2 3 4}); GPUS=(${GPUS:-1 3 5 7 6})
opts() { case $1 in 0|2|6) echo "--bs 4 --amp bf16";; *) echo "--bs 2 --accum 2 --amp fp16";; esac; }
for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}; g=${GPUS[$j]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 seg_pipeline.py train --fold $k --out runs/seg_f$k --cache $CACHE \
    $(opts $g) --workers 6 > runs/seg_f$k.out 2>&1 &
done
wait
for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}; g=${GPUS[$j]}
  CUDA_VISIBLE_DEVICES=$g python3 seg_pipeline.py predict --fold $k --out runs/seg_f$k --cache $CACHE \
    $(opts $g | grep -o -- "--amp [a-z0-9]*") --tta >> runs/seg_f$k.out 2>&1 &
done
wait
python3 seg_pipeline.py score --cache $CACHE --pred_dir runs/seg_oof/preds | tee runs/seg_oof/score.txt
