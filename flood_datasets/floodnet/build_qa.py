"""Usage: python flood_datasets/floodnet/build_qa.py
FloodNet in the shared schema (flood_datasets/common.py): the 4,511 answered QA pairs of `Training Question.json`
(val/test have no public answers), taken from vqa_full.csv so intents, normalised answers and the 5 image-level CV
folds (meta.fold) are exactly those of build_table.py. Original question wording is kept (paraphrases included).
Images point at the official re-download, because the old Images/ copy is truncated. The photos must be opened with
PIL.ImageOps.exif_transpose: 57 carry EXIF orientation 3, and the labels follow the rotated view."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, row, write_jsonl, summary

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = f"{ROOT}/FloodNet_official/FloodNet-Supervised_v1.0/train/train-org-img"
LAB = f"{ROOT}/FloodNet_official/FloodNet-Supervised_v1.0/train/train-label-img"

def main():
    v = pd.read_csv(f"{ROOT}/vqa_full.csv"); v = v[v.answer.notna()]
    rows, missing = [], 0
    for _, r in v.iterrows():
        iid = r.Image_ID.rsplit(".", 1)[0]; img = f"{IMG}/{iid}.jpg"
        if not os.path.exists(img): missing += 1; continue
        rows.append(row("floodnet", "train", iid, img, r.intent, r.answer, "human", r.Question,
                        fold=int(r.fold), label=os.path.relpath(f"{LAB}/{iid}_lab.png", ROOT)))
    if missing: print(f"WARNING: {missing} rows without an official image")
    write_jsonl(rows, f"{HERE}/qa.jsonl"); summary(rows)

if __name__ == "__main__":
    main()
