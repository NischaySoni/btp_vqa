# Flood datasets beyond FloodNet

Every dataset here is converted to one QA format, so the same answering pipeline and metrics run on all of them.
The format is defined in `common.py`: one JSON object per line in `<dataset>/qa.jsonl`.

| field | meaning |
|---|---|
| `dataset`, `split`, `image_id`, `image` | `image` is a path relative to the `btp_vqa/` root |
| `question`, `question_type` | FloodNet types (`Simple_Counting`, `Complex_Counting`, `Yes_No`, `Condition_Recognition`) plus `Scene_Recognition` and `Flood_Extent` |
| `intent` | Canonical question id. FloodNet intents keep the names from `build_table.py` (`count_total`, `count_flooded`, `overall_condition`, …); new ones are listed in `common.INTENTS` |
| `answer` | Normalised like `build_table.norm`: lower case, digits → int |
| `answer_source` | `human` (annotated), `label` (from image-level labels), `mask` (from a segmentation mask), `derived` (arithmetic on human answers) |
| `meta` | Provenance: event, country, source URL, licence, … |

Report accuracy separately per `answer_source`. Only `human` and `label` answers are ground truth in the usual sense.

## Status (2026-10-13)

| Dataset | Role | Images | QA pairs | Answers | Build |
|---|---|---|---|---|---|
| **FloodNet** (official images, answered train split) | main benchmark, 5-fold CV via `meta.fold` | 1,445 | 4,497 | human (original FloodNet answers, from `vqa_full.csv`) | `python flood_datasets/floodnet/build_qa.py` |
| **AIFloodSense** (global aerial, 64 countries, 2022–24) | cross-dataset test | 470 | 3,644 | 5 official human-verified questions on 294 images; flood extent from masks; sky / environment labels | `python flood_datasets/aifloodsense/build_qa.py` |
| **LADI v2** val + test (CAP aircraft, oblique, US) | cross-dataset test, flood vs ordinary water | 494 (247 flooded, 123 hard negatives with ordinary water, 124 other negatives) | 3,952 | FEMA-trained annotators' labels, yes/no and condition only | `python flood_datasets/ladi_v2/build_qa.py` |
| **web_real** (ours, Wikimedia Commons, real photos) | own real dataset | 721 candidates ready to annotate (760 downloaded, 39 near-duplicates / non-photos removed) | after annotation | human, 2 annotators | `collect_commons.py` → `--download` → `dedupe.py` → `annotate.py` → `build_qa.py` |
| **ai_generated** (ours, from Wikinews articles) | own synthetic dataset, robustness / failure analysis | 66 articles → 68 scene specs (`make_specs.py --scenes scenes_claude.json`); 136 images pending a GPU | after human check | intended by design, verified by humans | `fetch_news.py` → `make_specs.py` → `generate.py` → `annotate.py --dir flood_datasets/ai_generated` |

## Running the project

Run commands from the repository root. The main experiment is FloodNet: prepare the shared QA file, prepare official images and labels for segmentation, produce out-of-fold masks, then evaluate answers. Dataset-specific setup and annotation workflows are described below.

### 1. Prepare FloodNet QA and segmentation inputs

`vqa_full.csv` and `flood_datasets/floodnet/qa.jsonl` are already present. Regenerate them if needed:

```bash
python build_table.py . vqa_full.csv
python flood_datasets/floodnet/build_qa.py
```

Use the official FloodNet archive for segmentation inputs; the older `Images/` copy is truncated. After extracting the archive under `FloodNet_official/`, prepare the cache:

```bash
python seg_pipeline.py prep \
  --root FloodNet_official/FloodNet-Supervised_v1.0 \
  --vqa vqa_full.csv \
  --cache seg_cache_official
```

### 2. Train and predict the five FloodNet folds

The fold models must predict only images excluded from their own training data. The helper script trains folds 0, 1, 3, and 4 in parallel, then predicts their held-out images only if every training job succeeded and saved a checkpoint. Set `GPUS` to four available device IDs on this machine; `CACHE` overrides its machine-specific default:

```bash
CACHE=seg_cache_official FOLDS="0 1 3 4" GPUS="1 3 5 7" ./run_seg_folds.sh
```

The first run checks the masks for those four folds and defers final pooled scoring while fold 2 is missing. Train and predict fold 2 separately:

```bash
CACHE=seg_cache_official FOLDS="2" GPUS="0" ./run_seg_folds.sh
```

The script writes checkpoints under `runs/seg_f<fold>/` and held-out masks under `runs/seg_oof/preds/`. After the second run, it verifies all five folds cover the 1,445 expected images and writes the final pooled score. The all-data checkpoint in `runs/seg_all/` is for cross-dataset experiments; it has seen FloodNet VQA images and must not be used for FloodNet held-out scoring.

### 3. Evaluate FloodNet answers

Run the scene-to-answer rules on the out-of-fold masks. The tool fits its parameters within each training fold and scores held-out rows:

```bash
python answering/run_masks.py floodnet --labels runs/seg_oof/preds --fit own
python answering/evaluate.py floodnet answering/out/floodnet/masks_preds_own.jsonl --by intent
python answering/failure_gallery.py floodnet count_flooded \
  --preds answering/out/floodnet/masks_preds_own.jsonl
```

The blind prior baseline is available with `python answering/baselines.py`. Ground-truth-mask results (`--labels gt`) are an oracle comparison, not predicted-mask performance.

### 4. Run cross-dataset evaluations

The `runs/seg_all/` model is used to predict masks for datasets outside FloodNet. Prediction runs on CPU and can take time:

```bash
python answering/predict_cpu.py aifloodsense
python answering/run_masks.py aifloodsense \
  --labels answering/out/aifloodsense/preds_seg_all --fit floodnet

python answering/predict_cpu.py ladi_v2
python answering/run_masks.py ladi_v2 \
  --labels answering/out/ladi_v2/preds_seg_all --fit own
```

To rebuild external-dataset QA files, first place each dataset in the raw directory described by its builder, then run:

```bash
python flood_datasets/aifloodsense/build_qa.py
python flood_datasets/ladi_v2/build_qa.py
```

The real-photo annotation and AI-generated image workflows are in the sections below. Their QA builders write to the same schema, so use `answering/run_masks.py` and `answering/evaluate.py` where the dataset has compatible masks and labels.

## Results comparison

Results below are the latest completed scores recorded in `answering/README.md` (2026-10-29). Accuracy is percentage. Count scores are **exact / within ±1**. “Oracle” uses ground-truth masks, so it is an upper bound for perception rather than an end-to-end image result. FloodNet's five-fold out-of-fold run is complete: 1,445 masks, pooled segmentation mIoU 72.01, and end-to-end QA accuracy 80.2%.

| Dataset and evaluation split | Coverage | Blind baseline | Ground-truth-mask oracle | Predicted-mask pipeline | Readout |
|---|---|---|---|---|---|
| **FloodNet** — 5-fold CV | 4,497 QA / 1,445 images | Overall **64.0**; simple count 15.3 / 38.9; complex count 17.9 / 44.6 | Overall **92.2**; simple count 76.5 / 91.6; complex count 77.8 / 93.5; yes/no 99.7; condition 97.9 | Overall **80.2**; simple count 39.8 / 67.9; complex count 37.4 / 64.0; yes/no 98.7; condition 97.1. Pooled mask mIoU: 72.01. | OOF predictions across all five folds; published Kane & Khose reference: 81.3 overall (different split). |
| **AIFloodSense** — test | 43 VQA images plus 99 label rows; 3,644 QA overall | Overall 66.5; total count 27.9 / 27.9 | Overall **79.0**; total count 27.9 / 48.8 | Zero-shot: overall 51.1, total count 25.6 / 34.9. Calibrated: overall 65.1, total count 30.2 / 39.5 | The test VQA images are all labelled flooded: blind and calibrated flood yes/no reach 100%; zero-shot predicted masks reach 27.9%. |
| **LADI v2** — sampled test | 82 flooded, 41 ordinary-water hard negatives, 41 dry images; 3,952 QA overall | Flood yes/no 50.0 | Not available — no segmentation masks | Flood yes/no: **57.9** zero-shot, **69.5** calibrated. Flooded-image recall / hard-negative accuracy: 24.4 / 90.2 zero-shot; 81.7 / 36.6 calibrated. | Calibration improves flood recall but confuses ordinary water with flood. |
| **web_real** — candidate set | 721 deduplicated candidates; annotation not completed | — | — | — | No QA evaluation results yet. |
| **ai_generated** — planned robustness set | 68 scene specs from 66 articles; images not generated | — | — | — | No human-verified images or QA evaluation results yet. |

The FloodNet blind/oracle/predicted comparison is now complete. Flooded-building counting remains the main weakness: OOF accuracy is 14.8% exact / 36.2% within ±1, compared with 63.1% / 92.6% using ground-truth masks. The cross-dataset rows show substantial domain shift; web_real annotation and ai_generated image generation are still pending.

## Sources and licences

- **AIFloodSense:** CC BY 4.0 (dataset README). Download: Google Drive file `1dYl8aeSi_HbRFDbELEM8FWyg6zOdqLRp`, also on Kaggle (`cpanag/aifloodsense`). Cite Simantiris et al., *Remote Sensing* 2026 / arXiv 2512.17432. Its images were collected from news websites (URLs are in `meta.source_url`), so copyright of the individual photos stays with their owners. Use them for research evaluation only.
- **LADI v2:** Hugging Face `MITLL/LADI-v2-dataset` (v2a labels). Only the validation and test splits are downloaded (5.6 GB of 29 GB). Civil Air Patrol imagery. Cite Scheele et al., CVPRW 2025.
- **web_real:** Wikimedia Commons, keeping only public domain, CC0, CC BY and CC BY-SA files. Author, licence and page URL are kept per image in `candidates.csv`; attribution is required when images are shown.
- **ai_generated:** article text from Wikinews (CC BY 2.5). Images will be produced locally with an openly licensed model.

## Known issues

- **AIFloodSense:**
  - 5 VQA-test images (033, 277, 278, 411, 412) live in `train/images`.
  - 81 answers had a stray newline (`"\nyes"`); this is cleaned.
  - 3 images answer "Is the image flooded?" with "no".
  - 3 images have count 0 while the mask shows buildings (275: 14.5% building pixels, 277, 457).
- **AIFloodSense counts:** the authors dropped images with more than 80 buildings and report weak inter-annotator agreement on dense scenes. Treat counts as noisy.
- **LADI:** labels are image-level, so there are no counting questions yet. A hand-counted subset, done with `web_real/annotate.py`, is planned. `building_flooded_yn` comes from LADI's `flooding_structures` label.
- **web_real:** the Hurricane Harvey images come from the same event as FloodNet. `meta.found_via` lets you report them separately.

## Annotating web_real

```bash
python flood_datasets/web_real/annotate.py --port 8765    # then forward port 8765 (VS Code: Ports tab)
# open http://localhost:8765/?annotator=<name> ; two people annotate independently
python flood_datasets/web_real/build_qa.py                 # agreement report + qa.jsonl + disagreements.csv
```

Guidelines:
- Set `keep=no` for duplicates, non-flood photos, maps, graphics or people-only shots.
- Count a building only if its roof or walls are visible.
- A building counts as "flooded" if water touches it.
- Use `many` when a count would exceed 50.

## AI-generated set: remaining steps (need a working GPU)

```bash
CUDA_VISIBLE_DEVICES=6 python flood_datasets/ai_generated/make_specs.py              # Qwen2.5-7B-Instruct, ~66 articles -> up to 3 scenes each
CUDA_VISIBLE_DEVICES=6 /media/data_dump/Anwar/btp_gen_venv/bin/python flood_datasets/ai_generated/generate.py --seeds 2   # SDXL, 2 images/spec
python flood_datasets/web_real/annotate.py --dir flood_datasets/ai_generated --port 8766          # humans check each image
```

- **Spec design:** the article supplies place, setting, building type and the look of the water. Viewpoint, building count (1–8), number flooded, road and sky are sampled with a fixed seed, so answers are balanced. Answer-carrying phrases come first in every prompt, and prompts are trimmed to SDXL's 77-token text window.
- **Human check:** the annotator records what the image *actually* shows. The gap between intended and actual answers (e.g. the generator drew 6 houses instead of 8) is itself a result for the failure analysis.
- **CPU fallback:** `make_specs.py` falls back to CPU (float32, all cores) but takes about 15 min per article, so use a GPU.
