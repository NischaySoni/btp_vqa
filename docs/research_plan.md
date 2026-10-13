# Flood VQA: related work, review of current work, datasets, and proposed plan

Status: plan agreed 2026-10-13. Approach = segment → scene description → answer; Sen1Floods11 replaced by AIFloodSense. Step 1 (out-of-fold segmenter retrain, `seg_pipeline.py`) in progress.

---

## 1. Related work

### 1.1 Summary table

| # | Paper | Purpose | Issue #1 → how addressed | Issue #2 → how addressed | Use for FloodNet? (reason) |
|---|---|---|---|---|---|
| 1 | **FloodNet**, Rahnemoonfar et al., IEEE Access 2021 *(uploaded)* | Post-flood UAV dataset (Hurricane Harvey) for classification, segmentation, and VQA | No high-res UAV flood dataset existed (satellite = low res, slow revisit) → 2,343 UAV images (4000×3000), 10-class masks, ~4.5k QA pairs in 4 types | Need baselines → PSPNet 80.35 mIoU (flooded building 65.6); VQA: VGG16+LSTM at 224×224, no pretraining. **Counting only 0.21–0.32**, yes/no up to 0.98, condition 0.96–0.97 | **Yes, it's our benchmark.** Lesson: shrinking images to 224 px destroys small objects, and counting is the open problem |
| 2 | **VQA-Aid / HurMic-VQA**, Sarkar & Rahnemoonfar, IGARSS 2021 *(uploaded)* | UAV VQA for hurricane damage (Hurricane Michael) | No UAV VQA dataset for damage → 3,197 images with simple counting, complex counting (by damage level), and yes/no | Benchmark baselines → concat / point-wise / MFB with VGG+LSTM; overall only 0.55–0.60; complex counting lowest (0.53) | **Partly.** Reuse its question taxonomy (damage grades) for our own dataset. Confirms that end-to-end fusion models cannot count |
| 3 | **RSVQA**, Lobry et al., TGRS 2020 *(uploaded)* | Defines VQA for remote sensing | Manual QA labels are expensive → generate QA **automatically from OpenStreetMap** vectors using templates (presence, count, compare, area) | Large count answer space and imbalance → quantize counts into ranges (low-res set) | **Yes, the method.** Generate extra QA from FloodNet masks for the 895 val/test images that have no answers, and from Sen1Floods11 water masks |
| 4 | **Prompt-RSVQA**, Chappuis et al., CVPRW 2022 *(uploaded)* | Replace latent image/text fusion with text prompts | A fused embedding hides what the vision model actually "saw" → visual model predicts labels, labels become a **text prompt**, DistilBERT answers | Hard to know where errors come from → report a **visual-oracle** (GT labels) upper bound and a **visual-blind** (prior-only) lower bound; shows sensitivity to label noise | **Yes, closest to our idea.** Segment, convert to a structured text description, then answer. Adopt the oracle and blind bounds as ablations |
| 5 | **HRVQA**, Li, Vosselman & Yang, ISPRS J. 2024 *(uploaded)* | Large high-res aerial VQA benchmark (8 cm, NL) plus a model | Small, generic RS-VQA datasets → 53,512 images and 1.07M QA, **semi-automatic** from national geo-databases with attribute-rich templates (size, colour, 3×3 grid location) | Co-attention ignores position → **GFTransformer** (gated attention with positional embedding + mutual fusion). OA 81.7, but **Number questions only ~66%** even with 1M QA | **Partly.** Borrow the template/attribute design (location grid, size) for richer questions. The model itself is heavy and did not fix counting |
| 6 | **DisasterVQA**, Al-Mohannadi et al., AAAI 2026 (arXiv 2601.13839) *(uploaded)* | Ground-level disaster VQA grounded in FEMA/OCHA frameworks | No social-media disaster VQA → 1,395 images from CrisisMMD, Incidents1M, MEDIC. **GPT-4o generates QA, then two rounds of human verification**; adds hard negatives | Are VLMs reliable? → 7 VLMs zero-shot (GPT-4.1-mini best, binary 0.91). **Counting/quantitative questions are weakest** | **Yes, the protocol for our own dataset** (LLM-assisted QA + human verification + hard negatives). Its 289 flood images can serve as a cross-domain test |
| 7 | **Efficient Modern Baseline for FloodNet VQA**, Kane & Khose, arXiv 2205.15025 (2022) | Strong, cheap supervised baseline | Old encoders → **frozen ConvNeXt-L + RoBERTa-L**, element-wise product, 3 MLP layers. Overall **81.3%**, simple/complex counting **37.1/37.4%**, yes/no 98.3, condition 98.6/97.2 | Val/test answers are not public → own image-disjoint split (1,158 train / 290 test images) | **Yes, the main baseline to reproduce.** Same evaluation problem as ours |
| 8 | **SAM-VQA**, Sarkar et al., IEEE TGRS 2023 | Supervised-attention VQA for UAV damage | Attention looks at the wrong regions → **supervise attention with segmentation-derived masks** | Explainability → attention maps are interpretable. Counting ≈ 0.35 (as reported in ThiFAN Table III, extended FloodNet question set) | **Yes.** Evidence that masks help counting; we have masks for all 2,343 images |
| 9 | **Zero-shot flood VQA (ZFDDA) and VQA-TSP**, Sun, Wang & Peng, arXiv 2312.01882 / 2312.13848 | Zero-shot VQA for flood scenes using LLMs | No training data for a new disaster → captions plus **chain-of-thought** prompting of an LLM; builds FFD-IQA (2,058 images from CrisisMMD, FloodNet, European Flood 2013) | CoT hallucinates → **two-stage prompt**: reason first, then answer with visual context added | **Partly.** Relevant to the web-image datasets (free-form questions); not evaluated on counting |
| 10 | **ZeShot-VQA**, Karimi & Rahnemoonfar, IGARSS 2025 (arXiv 2506.00238) | Training-free VQA with an open answer space | Closed answer lists need retraining → BLIP generates freely, then **answer mapping** via CLIP-text similarity to the candidate answers | Density labels ambiguous → mapping ('scarce' → 'low'). Counting still 22–27%; **Qwen2-VL simple counting 9%** | **Partly.** Use answer mapping when the VLM branch answers in free text. Shows that zero-shot VLMs cannot count |
| 11 | **ThiFAN-VQA**, Karimi, Le & Rahnemoonfar, arXiv 2511.19557 | ICL + CoT VQA for FloodNet and RescueNet-VQA | Few labels; generic ICL → **question-type-aware retrieval** of exemplars + CoT | Hallucinated final answers → **answer selected from the reasoning trace only**. FloodNet overall 0.77, counting **0.27/0.30**; RescueNet-VQA counting 0.30/0.31 | **Partly.** Retrieval-augmented prompting for the VLM branch. Note: uses an *extended* FloodNet question set (building condition, density, risk) |
| 12 | **LADI v2**, Scheele, Picchione & Liu, CVPRW 2025 | Multi-label low-altitude disaster dataset | LADI v1 crowd labels were noisy and "damage" was undefined → ~10k Civil Air Patrol images labelled by **FEMA-trained volunteers** (12/15 labels incl. `flooding_any`, `flooding_structures`) | Real deployment shift → **test split = 2023 events** (new places/types) | **Yes, as a cross-dataset test** for yes/no and condition questions (§3) |
| 13 | **AIFloodSense**, Simantiris et al., Remote Sensing 2026 (arXiv 2512.17432) | Global aerial flood dataset | Flood data is US-centric → 470 images, 230 events, 64 countries; masks for flood/sky/building; VQA task | — | **Yes, a strong candidate cross-dataset test.** Aerial view, building masks, global. Licence CC BY-NC-ND |

Also cited by these papers and worth reading if time allows: Lobry et al., ISPRS Annals 2021 ("Better generic objects counting… multitask approach", which treats counting as **regression** instead of classification), and RescueNet / RescueNet-VQA (Hurricane Michael UAV: 4,375 images, 103k QA).

### 1.2 What the literature tells us

1. **Condition and yes/no questions are essentially solved on FloodNet** (96–99% for every modern method). Gains there are noise.
2. **Counting is the bottleneck in every paper**: 21–37% with supervised models, 9–30% with zero-shot/ICL VLMs, ~35% with mask-supervised attention, and ~66% even on HRVQA with 1M training QA.
3. The methods that help counting use **spatial supervision** (SAM-VQA masks; Prompt-RSVQA oracle labels). End-to-end fusion and VLM prompting do not.
4. Every FloodNet VQA paper has to make its own split, because val/test answers are not public. Our numbers need the same caveat.

---

## 2. Review of the existing work in this repo

### 2.1 What is there

| File | What it does | Verdict |
|---|---|---|
| `build_table.py` → `vqa_full.csv` | Parses the 3 question JSONs, maps each question to a canonical intent (7 intents), normalises answers, assigns **image-level 5-fold CV** over the 1,448 labelled images | Correct and the right design (only `Training Question.json` has answers: 4,511 QA) |
| `mask_count_check.py` → `mask_count_rows.csv` | Counts connected building regions in **GT** colour masks and compares them with count answers at several min-area thresholds | Valuable oracle study |
| `count_methods.py` → `count_preds_*.csv` | Turns GT masks into counts: prior / plain CC / watershed / ridge / GBR, with 5-fold CV | Good. Watershed over-splits; plain CC is best on exact match |
| `seg_pipeline.py` → `seg_cache/`, `runs/b2/` | SegFormer-B2 trained on the 898 masked val/test images (no VQA answers), predicts on the 1,445 VQA images | **Good leakage design, but two serious problems (below)** |
| `inspect_floodnet.py`, `path.py` | Folder inspection (`path.py` is an exact duplicate) | Fine |

### 2.2 Issues found (most serious first)

0. **Critical (found 2026-10-13, step 1): every JPEG in `Images/` is truncated.** Each file is exactly 2,359,296 or 2,621,440 bytes (a copy cut at 2.25/2.5 MiB). Only the top ~40–60% of each photo decodes; `LOAD_TRUNCATED_IMAGES = True` silently filled the rest with grey. So the old segmenter was trained to predict trees and buildings on grey pixels, and any image-based model sees half a scene. The masks and questions are unaffected. Fix: re-download the official FloodNet-Supervised_v1.0 zip (13 GB) into `FloodNet_official/`; `seg_pipeline.py` now refuses truncated JPEGs.
1. **Critical: the saved segmenter loads with a random decoder.** With transformers 5.5.3 the checkpoint is saved with keys `decode_head.linear_projections.*`, but the model expects `decode_head.linear_c.*`. `from_pretrained` silently re-initialises those layers. Running `seg_pipeline.py predict` as-is gives **5.98 mIoU** (it predicts "grass" everywhere). Fix: rename the keys when loading (done in the check below; not yet patched into `seg_pipeline.py`).
2. **Critical: the segmenter is weak even after the fix.** It scores **35.7 mIoU on seg_val** and **41.7 on the 1,445 VQA images** (flooded building 27.4), against 80.35 for PSPNet in the FloodNet paper. Likely causes:
   - only 809 training images;
   - ~4k iterations at lr 6e-5 (the SegFormer recipe uses 160k);
   - whole images shrunk ~4× to 1024×768, which removes small objects and building gaps;
   - the classifier weights stayed close to their initial values, so the model is under-trained.
3. **No training log** was saved, so we can't tell how many epochs ran (best checkpoint written about 32 min after the run started).
4. **Train/test mismatch in `count_methods.py`.** Ridge/GBR are fit on *GT-mask* features but will be applied to *predicted-mask* features. Fit them on out-of-fold predicted masks instead.
5. **Minor issues:**
   - The `split` column reads `training question.json` and `valid question.json` (the filename has no `_` to split on).
   - The `path` column holds Windows `F:\` paths that don't exist on the server.
   - 357 images are 4592×3072 and get squashed to 1024×768 (aspect distortion).
   - 8 road answers are `flooded,non flooded`; handle them explicitly.
   - The colour masks in `Masks/` are **1024×1024 for every image**: squashed over the full 4000×3000 (or 4592×3072) extent, not full resolution. The min-area thresholds in `mask_count_check.py` / `count_methods.py` labelled "full-res pixels" are therefore in mask pixels (1 mask px ≈ 3.9×2.9 image px).

### 2.3 Checks run now (5-fold CV over the 1,448 labelled images)

Scripts are in the session scratchpad; I can add them to the repo if wanted. Rule-based answers from masks: connected components (≥200 full-res px) for counting, and logistic regression on class-area fractions for condition and yes/no.

| Question type | From **GT** masks (oracle) | From **predicted** masks (current B2) | Kane & Khose 2022 (best, different split) |
|---|---|---|---|
| Simple counting | **62.2** | 18.2 | 37.1 |
| Complex counting | **62.7** | 18.3 | 37.4 |
| Yes/No | 97.0 | 97.8 | 98.3 |
| Condition recognition | 96.4 | 96.8 | 97.2–98.6 |
| Overall | 86.5 | 73.9 | 81.3 |

The oracle counting errors are **almost all under-counts in dense scenes**: adjacent buildings merge into one blob. Exact-match accuracy on total count from GT masks drops from 85% for ≤4 buildings to 47% for 5–10 and 16% for >10. For flooded buildings, 65% of answers are under-counts.

**Update (step 1, official labels).** The table above used the squashed 1024² colour masks, which ignore EXIF orientation for 57 photos. With the official full-resolution labels (EXIF-corrected), connected components with ≥600 px at the 2048 scale give these exact-match counting ceilings:

| Count | Exact | Within ±1 | Under-counts |
|---|---|---|---|
| Total | 75.8% | 90.7% | 21.6% |
| Non-flooded | 82.6% | 93.3% | 15.6% |
| Flooded | 58.4% | 84.6% | 37.6% |

Merging adjacent buildings is still the main residual error, mostly for flooded buildings.

**Conclusion:** the overall direction (segmentation → answers) is sound and beats the published counting results *when masks are good*. The current segmenter needs to be redone, and counting needs instance separation, not more fusion layers.

---

## 3. Dataset suitability (answer to the question to the advisor)

| Dataset | Suitable? | What we can use it for | Limits |
|---|---|---|---|
| **FloodNet-Supervised v1.0** | **Yes, main benchmark** | All 4 question types; masks for all 2,343 images | VQA answers only for 1,448 train images → 5-fold CV (or the 1,158/290 split from Kane & Khose for comparability). We can auto-generate condition/yes-no QA for the 895 val/test images from their masks (RSVQA-style) |
| **LADI v2** | **Yes, cross-dataset test (with limits)** | Yes/no and condition questions derived from labels: `flooding_any`, `flooding_structures`, `roads_damage`, `buildings_any`, … ; 2023-event test split is a real distribution shift | Image-level labels only: **no counting** unless we annotate a subset (e.g. 150 images with point clicks). Oblique views from manned aircraft |
| **Sen1Floods11** | **Only for a narrow sub-study** | "Is there flood water?", "What fraction is water?" from hand-labelled water masks (512×512 chips, Sentinel-1 SAR + Sentinel-2, 10 m) | At 10 m/px **buildings and roads are invisible**, so FloodNet's question types can't be asked. SAR needs its own input pipeline. Most useful as a documented *failure case* (§5, item 8) |
| **AIFloodSense** (suggested add) | **Yes** | Aerial, 64 countries, building + flood masks, VQA | Small (470 images); CC BY-NC-ND |
| **RescueNet / RescueNet-VQA** (suggested add) | Optional | UAV hurricane imagery with masks; tests transfer of the counting module | Hurricane damage, not flooding |
| **DisasterVQA flood subset** (suggested add) | Optional | 289 ground-level flood images with verified QA | Ground-level view; mostly non-counting questions |

---

## 4. Our own datasets (outputs 3 and 4)

**Real web images (non-AI).**
- **Size:** 200–300 images; aerial/UAV/oblique-aerial first, then some ground level.
- **Sources with usable licences:** Wikimedia Commons, Flickr CC, USGS/NOAA/FEMA public-domain imagery, Copernicus EMS.
- **Coverage:** several regions (e.g. Kerala 2018, Assam, Chennai 2015, Pakistan 2022, Libya 2023, US hurricanes).
- **Record per image:** URL, licence, event, date, and viewpoint.
- **Annotation:** point-click buildings (flooded / non-flooded), road condition, and overall condition. Two annotators with an agreement score.
- **Questions:** FloodNet's 4 types, plus the HRVQA-style location/size questions and DisasterVQA-style humanitarian questions (accessibility, need for rescue).

**AI-generated images.**
- **Prompts:** built from flood descriptions in news articles (place, water depth, building type, number of houses).
- **Answers:** taken from the prompt spec, then **human-verified**, because generators don't follow requested counts.
- **Role:** a robustness and failure-analysis set, never training data for the main results.

---

## 5. Proposed methodology (for discussion before implementing)

**Segment → structured scene description → answer**, with a VLM fallback for questions outside the templates.

1. **Question router.** Map each question to intent + target class + attribute. FloodNet templates are fixed (rules are 100% accurate); for free-form questions on the new datasets, use a small LLM.
2. **Perception (main work).**
   - A strong semantic segmenter trained on all 2,343 masks in k-fold fashion (no leakage into VQA test folds), using high-res crops. Candidates: SegFormer-B5 or UperNet-ConvNeXt.
   - **Building instance separation** (the oracle shows this is the counting bottleneck). Options:
     - (a) a boundary-aware head (interior / boundary / background);
     - (b) SAM/SAM2 automatic masks intersected with building classes;
     - (c) a point/density counting head.
3. **Reasoning.** For template questions, deterministic answers from the scene description. For open questions, prompt a VLM (e.g. Qwen2.5-VL-7B) with the image *plus* the structured description (Prompt-RSVQA idea), then apply answer mapping (ZeShot idea).
4. **Baselines.**
   - Answer prior (blind);
   - Kane & Khose ConvNeXt-mul reproduction;
   - zero-shot VLM;
   - optionally a LoRA-fine-tuned VLM.
5. **Ablations (output 7).**
   - GT vs predicted masks (oracle gap);
   - segmenter backbone and resolution;
   - instance-separation method;
   - count from components vs regression vs VLM;
   - VLM with vs without the scene description;
   - number of segmentation training images.
6. **Failure analysis (output 8).**
   - dense neighbourhoods and touching roofs;
   - flooded vs non-flooded buildings under tree cover;
   - oblique viewpoints (LADI, web images);
   - 10 m satellite (Sen1Floods11);
   - AI-generated images with impossible geometry.

**Metrics.** Accuracy per question type, plus within-±1 and MAE for counting, reported with the 5-fold CV and also on the 1,158/290 split for comparison with the literature.

---

## 6. Sources (beyond the uploaded PDFs)

- Kane & Khose 2022: https://arxiv.org/abs/2205.15025
- SAM-VQA (TGRS 2023): https://mdsoar.org/handle/11603/28136
- Sun, Wang, Peng: https://arxiv.org/abs/2312.01882 , https://arxiv.org/abs/2312.13848
- ZeShot-VQA: https://arxiv.org/abs/2506.00238
- ThiFAN-VQA: https://arxiv.org/abs/2511.19557
- LADI v2: https://arxiv.org/abs/2406.02780 , https://github.com/LADI-Dataset/ladi-overview
- AIFloodSense: https://arxiv.org/abs/2512.17432
- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11
- FloodNet: https://github.com/BinaLab/FloodNet-Supervised_v1.0
- DisasterBench (UAV multimodal disaster benchmark, 2026): https://arxiv.org/abs/2606.06217
