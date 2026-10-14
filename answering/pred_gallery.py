"""Usage: python answering/pred_gallery.py <dataset> [--preds answering/out/<dataset>/preds_seg_all] [--ids a,b,c] [--n 6]
Side-by-side photo | predicted FloodNet-class mask (seg_pipeline palette), with a legend: shows how the segmenter
behaves outside FloodNet's nadir UAV domain (failure analysis). Default picks --n evenly spaced images.
Writes answering/out/<dataset>/pred_gallery_<preds tag>.jpg."""
import os, sys, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from seg_pipeline import PAL, CLASSES

W, H = 420, 300

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--preds", default="")
    ap.add_argument("--ids", default=""); ap.add_argument("--n", type=int, default=6); a = ap.parse_args()
    pdir = a.preds or f"{ROOT}/answering/out/{a.dataset}/preds_seg_all"
    imgs = dict(sorted({(r["image_id"], r["image"]) for r in map(json.loads, open(f"{ROOT}/flood_datasets/{a.dataset}/qa.jsonl"))}))
    ids = a.ids.split(",") if a.ids else [i for i in imgs if os.path.exists(f"{pdir}/{i}.png")]
    ids = ids if a.ids else ids[:: max(1, len(ids) // a.n)][: a.n]
    sheet = Image.new("RGB", (2 * W + 10, len(ids) * (H + 6) + 40), "white"); d = ImageDraw.Draw(sheet)
    for k, i in enumerate(ids):
        im = ImageOps.exif_transpose(Image.open(f"{ROOT}/{imgs[i]}")).convert("RGB").resize((W, H))
        p = np.asarray(Image.open(f"{pdir}/{i}.png"))
        sheet.paste(im, (0, k * (H + 6))); sheet.paste(Image.fromarray(PAL[p].astype(np.uint8)).resize((W, H), Image.NEAREST), (W + 10, k * (H + 6)))
        d.text((4, k * (H + 6) + 4), i, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    y = len(ids) * (H + 6) + 8
    for c, (name, col) in enumerate(zip(CLASSES, PAL)):
        x = 6 + c * 84; d.rectangle((x, y, x + 12, y + 12), fill=tuple(int(v) for v in col)); d.text((x + 15, y), name[:12], fill=(0, 0, 0))
    out = f"{ROOT}/answering/out/{a.dataset}/pred_gallery_{os.path.basename(os.path.normpath(pdir))}.jpg"
    sheet.save(out, quality=88); print(out)

if __name__ == "__main__":
    main()
