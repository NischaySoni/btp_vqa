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
| **AIFloodSense** (global aerial, 64 countries, 2022–24) | cross-dataset test | 470 | 3,644 | 5 official human-verified questions on 294 images; flood extent from masks; sky / environment labels | `python flood_datasets/aifloodsense/build_qa.py` |
| **LADI v2** val + test (CAP aircraft, oblique, US) | cross-dataset test, flood vs ordinary water | 494 (247 flooded, 123 hard negatives with ordinary water, 124 other negatives) | 3,952 | FEMA-trained annotators' labels, yes/no and condition only | `python flood_datasets/ladi_v2/build_qa.py` |
| **web_real** (ours, Wikimedia Commons, real photos) | own real dataset | 721 candidates ready to annotate (760 downloaded, 39 near-duplicates / non-photos removed) | after annotation | human, 2 annotators | `collect_commons.py` → `--download` → `dedupe.py` → `annotate.py` → `build_qa.py` |
| **ai_generated** (ours, from Wikinews articles) | own synthetic dataset, robustness / failure analysis | 66 articles fetched; specs + images pending a GPU | after human check | intended by design, verified by humans | `fetch_news.py` → `make_specs.py` → `generate.py` → `annotate.py --dir flood_datasets/ai_generated` |

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

GPU 6 (A6000) is reserved for these steps. GPUs 0/1/3/5/7 run the segmenter folds, GPU 2 belongs to another user and GPU 4 is faulted. The shell profile sets `CUDA_VISIBLE_DEVICES=0`, so always override it.

```bash
CUDA_VISIBLE_DEVICES=6 python flood_datasets/ai_generated/make_specs.py              # Qwen2.5-7B-Instruct, ~66 articles -> up to 3 scenes each
CUDA_VISIBLE_DEVICES=6 /media/data_dump/Anwar/btp_gen_venv/bin/python flood_datasets/ai_generated/generate.py --seeds 2   # SDXL, 2 images/spec
python flood_datasets/web_real/annotate.py --dir flood_datasets/ai_generated --port 8766          # humans check each image
```

- **Spec design:** the article supplies place, setting, building type and the look of the water. Viewpoint, building count (1–8), number flooded, road and sky are sampled with a fixed seed, so answers are balanced. Answer-carrying phrases come first in every prompt, and prompts are trimmed to SDXL's 77-token text window.
- **Human check:** the annotator records what the image *actually* shows. The gap between intended and actual answers (e.g. the generator drew 6 houses instead of 8) is itself a result for the failure analysis.
- **CPU fallback:** `make_specs.py` falls back to CPU (float32, all cores) but takes about 15 min per article, so use a GPU.
