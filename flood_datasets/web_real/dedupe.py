"""Usage: python flood_datasets/web_real/dedupe.py [--max_dist 30]
Removes near-duplicates from candidates.csv before annotation: consecutive frames of one survey flight (common in the
GSI uploads) and any photo that also appears in AIFloodSense (so our set stays disjoint from it). Uses a 16x16
difference hash (256 bits); pairs closer than --max_dist bits count as the same scene (calibrated by eye: all pairs
< 25 were true duplicates, the first different-scene pair was at 37). Also drops obvious non-photos by title
(satellite / MODIS / radar / maps). Keeps the first of each group;
dropped rows go to candidates_dropped.csv with the id they duplicate. Original list is kept as candidates_all.csv."""
import os, re, csv, glob, shutil, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NOT_PHOTO = re.compile(r"MODIS|satellite|radar|\bmap\b|diagram|chart|Tropical Cyclone", re.I)
AIF = os.path.join(os.path.dirname(HERE), "aifloodsense", "raw", "AIFloodSense")

def dhash(p, n=16):
    im = Image.open(p); im.draft("L", (n * 8, n * 8))
    a = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max_dist", type=int, default=30); a = ap.parse_args()
    src = f"{HERE}/candidates_all.csv"
    if not os.path.exists(src): shutil.copy(f"{HERE}/candidates.csv", src)
    rows = [r for r in csv.DictReader(open(src)) if os.path.exists(f"{HERE}/images/{r['id']}.jpg")]
    H = np.stack([dhash(f"{HERE}/images/{r['id']}.jpg") for r in rows])
    aif = sorted(glob.glob(f"{AIF}/*/images/*.jpg")); A = np.stack([dhash(p) for p in aif]) if aif else None
    keep, drop = [], []
    for i, r in enumerate(rows):
        if NOT_PHOTO.search(r["title"]): drop.append(dict(r, dup_of="not a photo (title)")); continue
        if A is not None and (d := (A != H[i]).sum(1)).min() < a.max_dist:
            drop.append(dict(r, dup_of="aifloodsense:" + os.path.basename(aif[d.argmin()]))); continue
        near = [j for j in keep if (H[j] != H[i]).sum() < a.max_dist]
        if near: drop.append(dict(r, dup_of=rows[near[0]]["id"])); continue
        keep.append(i)
    with open(f"{HERE}/candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows[i] for i in keep)
    if drop:
        with open(f"{HERE}/candidates_dropped.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(drop[0])); w.writeheader(); w.writerows(drop)
    print(f"{len(rows)} downloaded -> {len(keep)} kept, {len(drop)} near-duplicates "
          f"({sum(d['dup_of'].startswith('aifloodsense') for d in drop)} also in AIFloodSense)")

if __name__ == "__main__":
    main()
