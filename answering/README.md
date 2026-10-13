# Answering stage: masks → scene description → answers

This is the reasoning half of the agreed methodology (`docs/research_plan.md` §5). The segmenter (`seg_pipeline.py`, owned by the step-1 session) produces label maps; this folder turns any label map into answers and scores them. Everything here is CPU-only.

| file | role |
|---|---|
| `evaluate.py` | Scores `{qid, pred}` predictions against `flood_datasets/<dataset>/qa.jsonl`. Reports accuracy per question type, intent, answer source or fold; counting also gets within-±1 and MAE |
| `baselines.py` | Blind baselines: the most frequent training answer per intent (lower bound) |
| `scene.py` | Label map → scene description: class shares, plus building instances, each flagged flooded or not |
| `reason.py` | One readable rule per intent, with one parameter fitted on training rows only (a threshold, or a min area plus optional merged-blob splitting for counts) |
| `run_masks.py` | Runs scene + reason on ground-truth labels (`--labels gt`, the oracle) or on a directory of predicted labels (e.g. `runs/seg_oof/preds`) |
| `failure_gallery.py` | Contact sheet of the worst counting errors, with every counted building outlined and numbered |

Protocol: FloodNet uses the 5 image-level CV folds of `vqa_full.csv`, fitting on 4 folds and scoring the 5th. AIFloodSense fits on its train split and scores its test split. LADI has no masks, so it waits for the segmenter. Rows whose answers were themselves derived from the dataset's masks (`answer_source = mask`) are excluded from mask-based scoring, since scoring them would be circular.

## Results so far (2026-10-13)

Accuracy in %. Counting cells are exact / within ±1.

| FloodNet (5-fold CV, 4,497 QA) | Blind prior | **Oracle: GT masks → rules** | Kane & Khose 2022 (best published, different split) |
|---|---|---|---|
| Simple counting | 15.3 / 38.9 | **76.5 / 91.6** | 37.1 |
| Complex counting | 17.9 / 44.6 | **77.8 / 93.5** | 37.4 |
| Yes/No | 82.0 | **99.7** | 98.3 |
| Condition recognition | 84.4 | **97.9** | 97.2–98.6 |
| Overall | 64.0 | **92.2** | 81.3 |

Counting by intent (oracle, exact / ±1): total 76.5 / 91.6, non-flooded 81.9 / 93.7, flooded **63.1 / 92.6**.

Without blob splitting, flooded counting is 56.4 / 83.9. Splitting merged blobs (component > 1.5 × the 75th-percentile area → round(area / reference)) is selected on the training folds for flooded and total counts.

| AIFloodSense test (43 VQA images + 99 label rows) | Blind prior | Oracle: GT masks → rules |
|---|---|---|
| Count total (exact / ±1) | 27.9 / 27.9 | 27.9 / 48.8 |
| Count flooded | — | 34.9 / 44.2 |
| Count non-flooded | — | 65.1 / 76.7 |
| Yes/No (sky, buildings, flooded) | 75.0 | 97.8 |
| Environment rural/urban | 62.6 | 73.7 |

LADI v2 test, blind prior: yes/no 80.5, condition 50.0 (balanced by construction).

### What the numbers say

1. **Blind priors are high on FloodNet:** 84% condition and 82% yes/no without looking at the image. The ~97% that every method reports is only 13–15 points above blind; counting is where methods actually differ.
2. **With perfect masks, simple rules double the best published counting accuracy (76–78% vs 37%) and reach 92% overall.** The pipeline's ceiling is therefore set by segmentation quality, and the segmenter is where effort should go.
3. **Flooded-building counts are hardest** (63% even with GT masks). Flooded neighbourhoods are dense, and touching roofs merge into one component.
4. **On oblique global imagery (AIFloodSense), mask counting fails even with GT masks**: 28% total, the same as the blind prior. Causes from `failure_gallery.py` (`out/aifloodsense/fail_count_total_masks_gt.jpg`):
   - distant small buildings fall under the min-area filter (image 142: 7 counted vs 71 true);
   - rows of buildings merge into one region in oblique views (114: 22 vs 49; 321: 17 vs 59);
   - some masks omit buildings entirely (411: answer 15, no building pixels).

   Counting in oblique views needs image-based instance separation (SAM-style masks or a point/density counting head), not semantic masks alone. This is a documented drawback for output 8.
5. **FloodNet's remaining oracle counting errors are small** (`out/floodnet/fail_count_flooded_masks_gt.jpg`): about 2–6 off in very dense scenes, mostly partial buildings at the image border and small outbuildings below the area threshold.

## Next (when the segmenter predictions exist)

```bash
python answering/run_masks.py floodnet --labels runs/seg_oof/preds    # full pipeline, same folds
python answering/failure_gallery.py floodnet count_flooded --preds answering/out/floodnet/masks_preds.jsonl
```

The gap between `masks_gt` and `masks_preds` is the segmentation error, and the gap between `masks_gt` and 100% is the reasoning plus label noise. Both feed the ablation table.
