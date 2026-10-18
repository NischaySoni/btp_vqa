#!/bin/bash
# Out-of-fold segmenters, then predictions for every VQA image (runs/seg_oof/preds) and the pooled mIoU.
#   FOLDS="0 1 3 4" GPUS="1 3 5 7" ./run_seg_folds.sh            (one GPU per fold, same order)
# 2080 Tis (1,3,5,7): fp16, batch 2 x accum 2; 3090/A6000 (0,2,6): bf16, batch 4. GPU 4 faulted on 2026-10-13.
# The all-data model (--fold -1, runs/seg_all) is trained separately on the 3090. GPU 6 (A6000) is left to the
# flood_datasets/ work (VLM + SDXL).
# The cache is read from local RAID (NAS reads were the bottleneck); rebuild it with `seg_pipeline.py prep` and copy.
set -uo pipefail

cd "$(dirname "$0")"
CACHE=${CACHE:-/media/data_dump/Anwar/btp_vqa_cache/seg_cache_official}
FOLDS=(${FOLDS:-0 1 3 4}); GPUS=(${GPUS:-1 3 5 7})   # fold 2: FOLDS=2 GPUS=0 once runs/seg_all frees the 3090
opts() { case $1 in 0|2|6) echo "--bs 4 --amp bf16";; *) echo "--bs 2 --accum 2 --amp fp16";; esac; }
amp() { case $1 in 0|2|6) echo bf16;; *) echo fp16;; esac; }

if [[ ! -f "$CACHE/split.json" ]]; then
  echo "Missing segmentation split: $CACHE/split.json (run seg_pipeline.py prep first)" >&2
  exit 1
fi
if (( ${#FOLDS[@]} != ${#GPUS[@]} )); then
  echo "FOLDS and GPUS must contain the same number of entries" >&2
  exit 1
fi
declare -A seen_folds
for k in "${FOLDS[@]}"; do
  if [[ ! "$k" =~ ^[0-4]$ ]]; then
    echo "Invalid fold '$k'; expected an integer from 0 to 4" >&2
    exit 1
  fi
  if [[ -n "${seen_folds[$k]:-}" ]]; then
    echo "Fold $k appears more than once" >&2
    exit 1
  fi
  seen_folds[$k]=1
done

mkdir -p runs/seg_oof/preds
train_pids=()
for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}; g=${GPUS[$j]}
  CUDA_VISIBLE_DEVICES=$g python3 seg_pipeline.py train --fold "$k" --out "runs/seg_f$k" --cache "$CACHE" \
    $(opts $g) --workers 6 > runs/seg_f$k.out 2>&1 &
  train_pids[$j]=$!
done

train_failed=0
for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}
  if wait "${train_pids[$j]}"; then
    echo "fold $k training finished"
  else
    echo "fold $k training failed; see runs/seg_f$k.out" >&2
    train_failed=1
  fi
done
if (( train_failed )); then
  echo "At least one training job failed; skipping prediction and scoring." >&2
  exit 1
fi

for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}; g=${GPUS[$j]}
  if [[ ! -s "runs/seg_f$k/best.safetensors" ]]; then
    echo "Missing checkpoint runs/seg_f$k/best.safetensors; skipping prediction." >&2
    exit 1
  fi
done

predict_pids=()
for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}; g=${GPUS[$j]}
  CUDA_VISIBLE_DEVICES=$g python3 seg_pipeline.py predict --fold "$k" --out "runs/seg_f$k" --cache "$CACHE" \
    --amp "$(amp "$g")" --tta >> "runs/seg_f$k.out" 2>&1 &
  predict_pids[$j]=$!
done

predict_failed=0
for j in "${!FOLDS[@]}"; do
  k=${FOLDS[$j]}
  if wait "${predict_pids[$j]}"; then
    echo "fold $k prediction finished"
  else
    echo "fold $k prediction failed; see runs/seg_f$k.out" >&2
    predict_failed=1
  fi
done
if (( predict_failed )); then
  echo "At least one prediction job failed; skipping pooled scoring." >&2
  exit 1
fi

python3 - "$CACHE/split.json" runs/seg_oof/preds "${FOLDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

split_path, pred_dir, *fold_args = sys.argv[1:]
vqa_folds = json.loads(Path(split_path).read_text())["vqa"]
selected_folds = {int(fold) for fold in fold_args}
expected = {image_id for image_id, fold in vqa_folds.items() if fold in selected_folds}
all_expected = set(vqa_folds)
found = {p.stem for p in Path(pred_dir).glob("*.png")}
missing_selected = sorted(expected - found)
extra = sorted(found - all_expected)
if missing_selected or extra:
    if missing_selected:
        print(f"Missing {len(missing_selected)} predictions for selected folds; examples: {missing_selected[:10]}", file=sys.stderr)
    if extra:
        print(f"Found {len(extra)} unexpected predictions; examples: {extra[:10]}", file=sys.stderr)
    sys.exit(1)
missing_all = sorted(all_expected - found)
print(f"OOF masks for selected folds: {len(expected)}/{len(expected)}")
if missing_all:
    pending_folds = sorted({vqa_folds[image_id] for image_id in missing_all})
    print(f"Final pooled score deferred: {len(missing_all)} images remain in folds {pending_folds}.")
    sys.exit(2)
print(f"OOF masks complete: {len(all_expected)}/{len(all_expected)} images")
PY
coverage_status=$?
if (( coverage_status == 2 )); then
  echo "Selected folds completed; run the remaining fold(s) to produce the final pooled score."
  exit 0
elif (( coverage_status != 0 )); then
  echo "OOF prediction coverage check failed; skipping pooled scoring." >&2
  exit 1
fi

python3 seg_pipeline.py score --cache "$CACHE" --pred_dir runs/seg_oof/preds | tee runs/seg_oof/score.txt
