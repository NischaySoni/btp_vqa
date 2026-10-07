"""Usage: python count_methods.py --vqa vqa_full.csv --masks F:\\btp_vqa\\Masks [--scale 4] [--limit 0] [--recompute]
Needs: pip install numpy pandas pillow scipy scikit-learn scikit-image
Compares, with image-level 5-fold CV (folds from vqa_full.csv), how to turn GROUND-TRUTH building masks
into count answers:  prior | plain region count | watershed split | ridge on region features | gradient boosting.
Features are cached in count_features.pkl (use --recompute to rebuild)."""
import os, glob, argparse, pickle
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None
BF, BNF = (255, 0, 0), (180, 120, 120)      # building flooded / non flooded (palette verified: 99.8% match)
MDS = [3, 6, 10]                            # watershed min_distance (pixels at working scale)
THR = 200                                   # min region area, full-res pixels
GRP = {"count_flooded": "fl", "count_nonflooded": "nf", "count_total": "all"}

def building_masks(path, scale, tol=30):
    im = Image.open(path).convert("RGB")
    im = im.resize((im.width // scale, im.height // scale), Image.NEAREST)
    a = np.asarray(im).astype(np.int32)
    f = lambda c: np.abs(a - np.array(c)).sum(2) <= tol
    return f(BF), f(BNF)

def ws_count(mask, md, scale):
    if not mask.any(): return 0
    dist = gaussian(ndi.distance_transform_edt(mask), 1)
    lab, _ = ndi.label(mask)
    pk = peak_local_max(dist, min_distance=md, labels=lab, exclude_border=False)
    if len(pk) == 0: return 0
    mk = np.zeros(mask.shape, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    ws = watershed(-dist, mk, mask=mask)
    areas = np.bincount(ws.ravel())[1:] * scale * scale
    return int((areas >= THR).sum())

def region_feats(mask, scale, pre):
    lab, n = ndi.label(mask)
    areas = np.bincount(lab.ravel())[1:] * scale * scale if n else np.array([0])
    d = {f"{pre}_n200": int((areas >= THR).sum()), f"{pre}_n1000": int((areas >= 1000).sum()),
         f"{pre}_nall": int(n), f"{pre}_frac": float(mask.mean()), f"{pre}_mean": float(areas.mean()),
         f"{pre}_med": float(np.median(areas)), f"{pre}_max": float(areas.max()), f"{pre}_std": float(areas.std())}
    for md in MDS: d[f"{pre}_ws{md}"] = ws_count(mask, md, scale)
    return d

def build_features(ids, mpath, scale):
    out = {}
    for k, i in enumerate(ids):
        fl, nf = building_masks(mpath[i], scale)
        d = {}
        for pre, m in (("fl", fl), ("nf", nf), ("all", fl | nf)): d.update(region_feats(m, scale, pre))
        out[i] = d
        if (k + 1) % 50 == 0: print(f"  features {k + 1}/{len(ids)}")
    return pd.DataFrame.from_dict(out, orient="index")

def metrics(y, p):
    e = np.abs(p - y)
    return dict(n=len(y), exact=(e == 0).mean(), within1=(e <= 1).mean(), MAE=e.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="vqa_full.csv"); ap.add_argument("--masks", required=True)
    ap.add_argument("--scale", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recompute", action="store_true"); a = ap.parse_args()

    df = pd.read_csv(a.vqa)
    df = df[df.intent.isin(GRP) & df.answer.notna()].copy()
    df["answer"] = pd.to_numeric(df.answer, errors="coerce"); df = df.dropna(subset=["answer"])
    df["id"] = df.Image_ID.str.rsplit(".", n=1).str[0]
    mpath = {os.path.basename(p)[:-8]: p for p in glob.glob(os.path.join(a.masks, "**", "*_lab.png"), recursive=True)}
    ids = sorted(i for i in df.id.unique() if i in mpath)
    if a.limit: ids = ids[:a.limit]
    cache = "count_features.pkl"
    if os.path.exists(cache) and not a.recompute and not a.limit:
        feats = pickle.load(open(cache, "rb"))
    else:
        feats = build_features(ids, mpath, a.scale)
        if not a.limit: pickle.dump(feats, open(cache, "wb"))
    df = df[df.id.isin(feats.index)]

    rows = []
    for intent, g in GRP.items():
        d = df[df.intent == intent].reset_index(drop=True)
        X = feats.loc[d.id].reset_index(drop=True); y = d.answer.values.astype(float)
        cols = [c for c in X.columns if c.startswith(g + "_")]
        pred = {m: np.zeros(len(d)) for m in ("prior", "plain", "ws_best", "ridge", "gbr")}
        for f in range(5):
            te = (d.fold == f).values; tr = ~te & d.fold.notna().values
            if te.sum() == 0: continue
            pred["prior"][te] = pd.Series(y[tr]).mode().iloc[0]
            pred["plain"][te] = X.loc[te, f"{g}_n200"]
            maes = {md: np.abs(X.loc[tr, f"{g}_ws{md}"] - y[tr]).mean() for md in MDS}
            pred["ws_best"][te] = X.loc[te, f"{g}_ws{min(maes, key=maes.get)}"]
            for name, mdl in (("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10))),
                              ("gbr", GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0))):
                mdl.fit(X.loc[tr, cols], y[tr]); pred[name][te] = np.clip(np.rint(mdl.predict(X.loc[te, cols])), 0, None)
        for m, p in pred.items():
            rows.append(dict(intent=intent, method=m, **metrics(y, p)))
            d[f"pred_{m}"] = p
        d.to_csv(f"count_preds_{intent}.csv", index=False)
        d["bin"] = pd.cut(d.answer, [-1, 4, 7, 1000], labels=["<=4", "5-7", ">=8"])
        print(f"\n=== {intent}: error by density bin (MAE / exact) ===")
        for m in pred:
            t = d.groupby("bin", observed=True).apply(lambda s: f"{(s[f'pred_{m}'] - s.answer).abs().mean():.2f}/{((s[f'pred_{m}'] - s.answer) == 0).mean():.2f}")
            print(f"  {m:8s}", dict(t))
    print("\n=== overall (5-fold CV by image) ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()