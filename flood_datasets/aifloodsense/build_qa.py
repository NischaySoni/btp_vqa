"""Usage: python flood_datasets/aifloodsense/build_qa.py   (expects flood_datasets/aifloodsense/raw/AIFloodSense from the zip)
Writes flood_datasets/aifloodsense/qa.jsonl in the shared schema (flood_datasets/common.py).
- 294 VQA images: the 5 official human-verified answers (question text kept verbatim) plus answers derived from them
  (non-flooded count = total - flooded, yes/no forms, overall condition).
- all 470 images: flood extent from the mask; sky / environment from the metadata labels for images outside the VQA set.
Note: 5 VQA-test images (033, 277, 278, 411, 412) sit in train/images; split follows the VQA csv, meta.seg_split
records the folder. We only evaluate on AIFloodSense, so the overlap does not leak into training."""
import os, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import row, write_jsonl, summary, extent_answer

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = f"{HERE}/raw/AIFloodSense"
FLOOD, SKY, BUILDING = 255, 170, 85            # grey values in the masks
Q = {"count_total": "How many buildings are in the image?", "environment": "What environment is flooded?",
     "sky_yn": "Is there sky present?", "image_flooded_yn": "Is the image flooded?",
     "count_flooded": "How many buildings are flooded?"}

def main():
    where = {f[:-4]: s for s in ("train", "test") for f in os.listdir(f"{RAW}/{s}/images")}
    meta = pd.concat([pd.read_csv(f"{RAW}/AIFloodSense-{s}.csv", dtype={"IMAGE_ID": str}).assign(split=s)
                      for s in ("train", "test")]).set_index("IMAGE_ID")
    vqa = pd.concat([pd.read_csv(f"{RAW}/AIFloodSense-VQA-{s}.csv").assign(split=s) for s in ("train", "test")])
    vqa["id"] = vqa["Img Name"].str[:-4]; vqa = vqa.set_index("id")
    rows = []
    for iid, seg_split in sorted(where.items()):
        img = f"{RAW}/{seg_split}/images/{iid}.jpg"
        m = meta.loc[iid]; split = vqa.loc[iid, "split"] if iid in vqa.index else seg_split
        info = dict(seg_split=seg_split, event=m.FLOOD_NAME, year=int(m.YEAR), country=m.COUNTRY,
                    continent=m.CONTINENT, source_url=m.SOURCE)
        R = lambda intent, ans, src, q=None: rows.append(row("aifloodsense", split, iid, img, intent, ans, src, q, **info))
        mask = np.asarray(Image.open(f"{RAW}/{seg_split}/masks/{iid}.png"))
        R("flood_extent", extent_answer((mask == FLOOD).mean()), "mask")
        if iid in vqa.index:
            v = vqa.loc[iid]; ans = {k: str(v[q]).strip() for k, q in Q.items()}
            for k, q in Q.items(): R(k, ans[k], "human", q)
            tot, fl = int(ans["count_total"]), int(ans["count_flooded"])
            R("count_nonflooded", tot - fl, "derived")
            R("building_any_yn", "yes" if tot else "no", "derived")
            R("building_flooded_yn", "yes" if fl else "no", "derived")
            R("overall_condition", "flooded" if ans["image_flooded_yn"].lower() == "yes" else "non flooded", "derived")
        else:
            R("sky_yn", "yes" if m.SKY == "With Sky" else "no", "label")
            R("environment", "rural" if m.ENVIRONMENT == "Rural" else "urban", "label")
            R("building_any_yn", "yes" if (mask == BUILDING).mean() >= 0.001 else "no", "mask")
    write_jsonl(rows, f"{HERE}/qa.jsonl"); summary(rows)

if __name__ == "__main__":
    main()
