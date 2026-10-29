# Answering stage: masks → scene description → answers

This is the reasoning half of the agreed methodology (`docs/research_plan.md` §5). The segmenter (`seg_pipeline.py`, owned by the step-1 session) produces label maps; this folder turns any label map into answers and scores them. Everything here is CPU-only.

| file | role |
|---|---|
| `evaluate.py` | Scores `{qid, pred}` predictions against `flood_datasets/<dataset>/qa.jsonl`. Reports accuracy per question type, intent, answer source or fold; counting also gets within-±1 and MAE |
| `baselines.py` | Blind baselines: the most frequent training answer per intent (lower bound) |
| `scene.py` | Label map → scene description: class shares, plus building instances, each flagged flooded or not |
| `reason.py` | One readable rule per intent, with one parameter fitted on training rows only (a threshold, or a min area plus optional merged-blob splitting for counts) |
| `run_masks.py` | Runs scene + reason on ground-truth labels (`--labels gt`, the oracle) or on a directory of predicted labels. `--fit own` calibrates on the dataset's own training rows; `--fit floodnet` uses FloodNet-fitted parameters unchanged (zero-shot transfer). Intents the classes can't express fall back to the blind prior and are flagged `how=prior` in the output |
| `predict_cpu.py` | Predicted FloodNet-class masks on CPU (about 15 s/image) with a trained segmenter. `runs/seg_all` is only for non-FloodNet datasets, because it trained on the FloodNet VQA images |
| `failure_gallery.py` | Contact sheet of the worst counting errors, with every counted building outlined and numbered |

Protocol: FloodNet uses the 5 image-level CV folds of `vqa_full.csv`, fitting on 4 folds and scoring the 5th. AIFloodSense fits on its train split and scores its test split. LADI has no segmentation masks, so its predicted-mask pipeline uses the FloodNet-trained segmenter. Rows whose answers were themselves derived from the dataset's masks (`answer_source = mask`) are excluded from mask-based scoring, since scoring them would be circular.

## Results so far (2026-10-29)

Accuracy in %. Counting cells are exact / within ±1.

| FloodNet (5-fold CV, 4,497 QA) | Blind prior | Oracle: GT masks → rules | **OOF predicted masks → rules** | Kane & Khose 2022 (different split) |
|---|---|---|---|---|
| Simple counting | 15.3 / 38.9 | 76.5 / 91.6 | **39.8 / 67.9** | 37.1 |
| Complex counting | 17.9 / 44.6 | 77.8 / 93.5 | **37.4 / 64.0** | 37.4 |
| Yes/No | 82.0 | 99.7 | **98.7** | 98.3 |
| Condition recognition | 84.4 | 97.9 | **97.1** | 97.2–98.6 |
| Overall | 64.0 | 92.2 | **80.2** | 81.3 |

The pooled OOF segmentation mIoU is **72.01** across all 1,445 VQA images. Per-fold QA accuracy: fold 0 79.1%, fold 1 80.8%, fold 2 81.3%, fold 3 80.0%, fold 4 79.9%.

Counting by intent (oracle, exact / ±1): total 76.5 / 91.6, non-flooded 81.9 / 93.7, flooded **63.1 / 92.6**.

Counting by intent (OOF predicted masks, exact / ±1): total 39.8 / 67.9, non-flooded 43.7 / 71.7, flooded **14.8 / 36.2**. The worst flooded-count cases are collected in `answering/out/floodnet/fail_count_flooded_masks_preds_own.jpg`.

Without blob splitting, flooded counting is 56.4 / 83.9. Splitting merged blobs (component > 1.5 × the 75th-percentile area → round(area / reference)) is selected on the training folds for flooded and total counts.

| AIFloodSense test (43 VQA images + 99 label rows) | Blind prior | Oracle: GT masks → rules (fit on AIF train) |
|---|---|---|
| Count total (exact / ±1) | 27.9 / 27.9 | 27.9 / 48.8 |
| Count flooded | — | 34.9 / 44.2 |
| Count non-flooded | — | 65.1 / 76.7 |
| Yes/No (sky, buildings, flooded) | 75.0 | 97.8 |
| Environment rural/urban | 62.6 | 73.7 |

With thresholds fitted on FloodNet only (zero-shot), the AIFloodSense oracle is essentially unchanged: 79.0% overall; count total 30.2 / 46.5, flooded 37.2 / 51.2. The rules transfer; the masks are the hard part.

LADI v2 test, blind prior: yes/no 80.5, condition 50.0 (balanced by construction).

### Full pipeline on the other datasets (predicted masks from `runs/seg_all`, CPU)

The segmenter was trained only on FloodNet (nadir UAV, Houston). Predictions come from `predict_cpu.py` at a long side of 2048 px. *Zero-shot* = rule parameters fitted on FloodNet GT; *calibrated* = fitted on the dataset's own train/validation split.

| AIFloodSense test | Blind | GT masks | Pred, zero-shot | Pred, calibrated |
|---|---|---|---|---|
| Is it flooded? (all test images are flooded) | 100 | 100 | **27.9** | 100 |
| Count total (exact / ±1) | 27.9 / 27.9 | 27.9 / 48.8 | 25.6 / 34.9 | 30.2 / 39.5 |
| Count flooded | — | 34.9 / 44.2 | 32.6 / 32.6 | 34.9 / 37.2 |
| Buildings present? | — | 93.0 | 90.7 | 90.7 |
| Overall (sky answered by the blind prior) | 66.5 | 79.0 | 51.1 | 65.1 |

| LADI v2 test (82 flooded / 41 ordinary water / 41 dry) | Blind | Pred, zero-shot | Pred, calibrated |
|---|---|---|---|
| Is it flooded? | 50.0 | 57.9 | **69.5** |
| — flooded images | | 24.4 | 81.7 |
| — ordinary water, no flood (hard negatives) | | 90.2 | **36.6** |
| — dry, with buildings | | 92.7 | 78.0 |
| Buildings present? | 73.8 | 80.5 | 80.5 |
| Any building flooded? | 92.7 | 87.8 | 91.5 |
| Road present? | 85.4 | 78.0 | 78.0 |
| Water present? | 75.0 | 76.8 | 76.8 |

Damage questions are answered by the blind prior, because FloodNet classes have no damage notion.

### What the numbers say

1. **Blind priors are high on FloodNet:** 84% condition and 82% yes/no without looking at the image. The ~97% that every method reports is only 13–15 points above blind; counting is where methods actually differ.
2. **With perfect masks, simple rules double the best published counting accuracy (76–78% vs 37%) and reach 92% overall.** The pipeline's ceiling is therefore set by segmentation quality, and the segmenter is where effort should go.
3. **Flooded-building counts are hardest** (63% even with GT masks). Flooded neighbourhoods are dense, and touching roofs merge into one component.
4. **On oblique global imagery (AIFloodSense), mask counting fails even with GT masks**: 28% total, the same as the blind prior. Causes from `failure_gallery.py` (`out/aifloodsense/fail_count_total_masks_gt.jpg`; regenerate with the new default `masks_gt_own`):
   - distant small buildings fall under the min-area filter (image 142: 7 counted vs 71 true);
   - rows of buildings merge into one region in oblique views (114: 22 vs 49; 321: 17 vs 59);
   - some masks omit buildings entirely (411: answer 15, no building pixels).

   Counting in oblique views needs image-based instance separation (SAM-style masks or a point/density counting head), not semantic masks alone. This is a documented drawback for output 8.
5. **Domain shift breaks the segmenter, and with it the pipeline** (`out/aifloodsense/pred_gallery_preds_seg_all.jpg`):
   - Oblique and global photos get flood water labelled as plain "water" or even dry road (411, 031).
   - Dark-roofed houses standing in water become "water" (114: 2 of ~30 buildings found).
   - Dense villages become background (002, 142).
   - Zero-shot flood detection therefore collapses (AIFloodSense 27.9%; LADI 24% of floods found).
   - Calibrating thresholds on the target data recovers flood recall (LADI 82%) but then confuses ordinary rivers and lakes with floods: only 37% of hard negatives are correct.

   **The 10 FloodNet classes cannot separate flood water from normal water outside FloodNet's setting.** Remedies to test:
   - fine-tune on AIFloodSense masks (flood vs background);
   - add a flood/no-flood image classifier as an extra feature (e.g. trained on LADI's `flooding_any`);
   - a VLM fallback for the condition questions.
6. **FloodNet's remaining oracle counting errors are small** (`out/floodnet/fail_count_flooded_masks_gt.jpg`): about 2–6 off in very dense scenes, mostly partial buildings at the image border and small outbuildings below the area threshold.

## Next steps

```bash
python answering/run_masks.py floodnet --labels runs/seg_oof/preds    # full pipeline, same folds
python answering/failure_gallery.py floodnet count_flooded --preds answering/out/floodnet/masks_preds_own.jsonl
# other datasets, with the all-data segmenter (CPU):
python answering/predict_cpu.py aifloodsense && python answering/run_masks.py aifloodsense --labels answering/out/aifloodsense/preds_seg_all --fit floodnet
python answering/predict_cpu.py ladi_v2 && python answering/run_masks.py ladi_v2 --labels answering/out/ladi_v2/preds_seg_all --fit own
```

FloodNet OOF predictions, scoring, and the flooded-count failure gallery are complete. Next, inspect that gallery and improve flooded-building counting; then continue the human annotation of `web_real` and generate and verify the AI-generated images. For cross-dataset runs, the AIFloodSense and LADI commands above reproduce the existing predicted-mask comparisons.
