"""Usage: python answering/failure_gallery.py <dataset> <intent> [--preds answering/out/<dataset>/masks_gt_own.jsonl] [--n 8]
Contact sheet of the worst counting errors (|pred - answer| largest) for the failure analysis: image with every
counted building component outlined (red = flooded, cyan = not flooded) and numbered, plus question / answer / pred.
Writes answering/out/<dataset>/fail_<intent>_<preds tag>.jpg."""
import os, sys, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_qa, ROOT

W = 520   # panel width

def panel(img_path, lab_path, kind, title):
    im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    lab = np.asarray(Image.open(lab_path))
    im = im.resize((lab.shape[1], lab.shape[0]))
    over = np.asarray(im).copy(); txt = []
    groups = [(lab == 1, (255, 40, 40)), (lab == 2, (0, 230, 255))] if kind == "floodnet" else [(lab == 85, (255, 200, 0))]
    if kind != "floodnet": over[lab == 255] = (0.5 * over[lab == 255] + [0, 60, 127]).astype(np.uint8)   # flood tint
    for m, col in groups:
        cc, n = ndi.label(m)
        edge = m & ~ndi.binary_erosion(m, iterations=max(2, lab.shape[1] // 400))
        over[edge] = col
        for k, (sy, sx) in enumerate(ndi.find_objects(cc), 1):
            txt.append(((sx.start + sx.stop) // 2, (sy.start + sy.stop) // 2, str(len(txt) + 1), col))
    p = Image.fromarray(over); s = W / p.width; p = p.resize((W, int(p.height * s)))
    d = ImageDraw.Draw(p)
    for x, y, t, col in txt: d.text((x * s - 4, y * s - 6), t, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    d.rectangle((0, 0, W, 34), fill=(0, 0, 0)); d.text((5, 3), title, fill=(255, 255, 0))
    return p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("intent")
    ap.add_argument("--preds", default=""); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
    preds_path = a.preds or f"{ROOT}/answering/out/{a.dataset}/masks_gt_own.jsonl"
    qa = load_qa(a.dataset); preds = {json.loads(l)["qid"]: json.loads(l)["pred"] for l in open(preds_path)}
    rows = [(abs(int(preds[i]) - r["answer"]), i) for i, r in enumerate(qa) if r["intent"] == a.intent and i in preds]
    rows.sort(reverse=True); panels = []
    for err, i in rows[: a.n]:
        r = qa[i]; iid = r["image_id"]
        if a.dataset == "floodnet":
            img, lab, kind = f"{ROOT}/seg_cache_official/img/{iid}.jpg", f"{ROOT}/seg_cache_official/lab/{iid}.png", "floodnet"
        else:
            img = f"{ROOT}/{r['image']}"; lab, kind = img.replace("/images/", "/masks/").replace(".jpg", ".png"), "aif"
        panels.append(panel(img, lab, kind, f"{iid}: {r['question']}\nanswer {r['answer']}  |  mask-count {preds[i]}"))
    h = max(p.height for p in panels); cols = 2; rws = (len(panels) + 1) // 2
    sheet = Image.new("RGB", (cols * (W + 6), rws * (h + 6)), "white")
    for k, p in enumerate(panels): sheet.paste(p, ((k % cols) * (W + 6), (k // cols) * (h + 6)))
    out = f"{ROOT}/answering/out/{a.dataset}/fail_{a.intent}_{os.path.basename(preds_path)[:-6]}.jpg"
    sheet.save(out, quality=88); print(out, f"({len(rows)} rows, worst errors {[e for e, _ in rows[: a.n]]})")

if __name__ == "__main__":
    main()
