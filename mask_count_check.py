"""Usage: python mask_count_check.py --vqa vqa_full.csv --masks F:\\btp_vqa\\Masks [--scale 2] [--tol 30] [--limit 0]
Needs: pip install numpy pandas pillow scipy
Counts connected building regions in the ground-truth colour masks and compares with
the count answers (count_total / count_flooded / count_nonflooded)."""
import os, glob, argparse
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
# Reference palette (ADE20K-style, from memory) -- the script reports match coverage so a wrong guess is visible.
REF = {"background": (0, 0, 0), "building_flooded": (255, 0, 0), "building_nonflooded": (180, 120, 120),
       "road_flooded": (160, 150, 20), "road_nonflooded": (140, 140, 140), "water": (61, 230, 250),
       "tree": (0, 82, 255), "vehicle": (255, 0, 245), "pool": (255, 235, 0), "grass": (4, 250, 7)}
NAMES = list(REF)
THRESH = [0, 200, 1000, 3000, 10000]   # min region area in FULL-resolution pixels

def classify(path, scale, tol):
    im = Image.open(path).convert("RGB")
    if scale > 1:
        im = im.resize((im.width // scale, im.height // scale), Image.NEAREST)
    a = np.asarray(im).astype(np.int32)
    packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    cols, inv = np.unique(packed.ravel(), return_inverse=True)
    counts = np.bincount(inv, minlength=len(cols))
    rgb = np.stack([(cols >> 16) & 255, (cols >> 8) & 255, cols & 255], 1)
    d = np.abs(rgb[:, None, :] - np.array(list(REF.values()))[None]).sum(2)
    best = d.argmin(1)
    ok = d[np.arange(len(cols)), best] <= tol
    lab = np.where(ok, best, -1)
    unmatched = {tuple(int(x) for x in rgb[i]): int(counts[i]) for i in np.where(~ok)[0]}
    return lab[inv].reshape(packed.shape), unmatched, int(counts[ok].sum()), int(counts.sum())

def n_regions(mask, scale):
    lab, n = ndi.label(mask)
    if n == 0: return {t: 0 for t in THRESH}
    areas = np.bincount(lab.ravel())[1:] * scale * scale   # back to full-res pixels
    return {t: int((areas >= t).sum()) for t in THRESH}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="vqa_full.csv"); ap.add_argument("--masks", required=True)
    ap.add_argument("--scale", type=int, default=2); ap.add_argument("--tol", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (quick test)")
    a = ap.parse_args()

    df = pd.read_csv(a.vqa)
    df = df[df.intent.str.startswith("count_") & df.answer.notna()].copy()
    df["answer"] = pd.to_numeric(df.answer, errors="coerce"); df = df.dropna(subset=["answer"])
    mpath = {os.path.basename(p)[:-8]: p for p in glob.glob(os.path.join(a.masks, "**", "*_lab.png"), recursive=True)}
    df["id"] = df.Image_ID.str.rsplit(".", n=1).str[0]
    imgs = sorted(df.id.unique())
    have = [i for i in imgs if i in mpath]
    print(f"labelled count images: {len(imgs)}   with mask: {len(have)}")
    if a.limit: have = have[:a.limit]

    per, unm, cov_hit, cov_all = {}, {}, 0, 0
    for k, i in enumerate(have):
        cls, um, hit, tot = classify(mpath[i], a.scale, a.tol)
        cov_hit += hit; cov_all += tot
        for c, n in um.items(): unm[c] = unm.get(c, 0) + n
        per[i] = (n_regions(cls == NAMES.index("building_flooded"), a.scale),
                  n_regions(cls == NAMES.index("building_nonflooded"), a.scale))
        if (k + 1) % 100 == 0: print(f"  processed {k + 1}/{len(have)}")
    print(f"\npixels matched to reference palette: {100 * cov_hit / max(cov_all, 1):.2f}%")
    if unm: print("top unmatched colours (RGB: pixels):", sorted(unm.items(), key=lambda x: -x[1])[:8])

    rows = []
    for _, r in df[df.id.isin(per)].iterrows():
        fl, nf = per[r.id]
        for t in THRESH:
            pred = {"count_flooded": fl[t], "count_nonflooded": nf[t], "count_total": fl[t] + nf[t]}[r.intent]
            rows.append((r.intent, t, r.Image_ID, r.answer, pred))
    res = pd.DataFrame(rows, columns=["intent", "min_area", "image", "answer", "pred"])
    res["err"] = (res.pred - res.answer).abs()
    res.to_csv("mask_count_rows.csv", index=False)
    summ = res.groupby(["intent", "min_area"]).agg(n=("err", "size"), exact=("err", lambda e: (e == 0).mean()),
                                                  within1=("err", lambda e: (e <= 1).mean()), MAE=("err", "mean")).round(3)
    print("\n", summ.to_string())

if __name__ == "__main__":
    main()