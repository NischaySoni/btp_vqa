"""Usage: python answering/predict_cpu.py <dataset> [--model runs/seg_all] [--long 2048] [--threads 24] [--limit 0]
Predicted FloodNet-class label maps for every image of flood_datasets/<dataset>/qa.jsonl, on CPU (the GPUs cannot start
new CUDA processes right now). Each image is EXIF-transposed, resized so its long side is --long (the scale the
segmenter was trained at), normalised like seg_pipeline, and segmented in fp32. Output: one class-id PNG per image in
answering/out/<dataset>/preds_<model tag>/<image_id>.png, at inference resolution.
runs/seg_all trained on the FloodNet VQA images, so use it only for the OTHER datasets; FloodNet needs the fold models."""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageOps
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import seg_pipeline as sp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--model", default="runs/seg_all")
    ap.add_argument("--long", type=int, default=2048); ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    os.sched_setaffinity(0, range(os.cpu_count())); torch.set_num_threads(a.threads)
    rows = [json.loads(l) for l in open(f"{ROOT}/flood_datasets/{a.dataset}/qa.jsonl")]
    imgs = sorted({(r["image_id"], r["image"]) for r in rows})[: a.limit or None]
    out = f"{ROOT}/answering/out/{a.dataset}/preds_{os.path.basename(os.path.normpath(a.model))}"
    os.makedirs(out, exist_ok=True)
    model = sp.load_trained(f"{ROOT}/{a.model}", "cpu").eval(); t0 = time.time()
    for k, (iid, rel) in enumerate(imgs):
        p = f"{out}/{iid}.png"
        if os.path.exists(p): continue
        im = ImageOps.exif_transpose(Image.open(f"{ROOT}/{rel}")).convert("RGB")
        s = a.long / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        x = (np.asarray(im, np.float32) / 255 - sp.MEAN) / sp.STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
        with torch.no_grad():
            lab = F.interpolate(model(pixel_values=x).logits, size=x.shape[-2:], mode="bilinear").argmax(1)[0]
        Image.fromarray(lab.numpy().astype(np.uint8)).save(p)
        if (k + 1) % 25 == 0: print(f"{k + 1}/{len(imgs)}  {(time.time() - t0) / (k + 1):.1f}s/img", flush=True)
    print(f"done: {len(imgs)} images -> {out}")

if __name__ == "__main__":
    main()
