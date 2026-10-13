"""FloodNet segmentation pipeline, out-of-fold. Needs: pip install torch transformers safetensors pillow numpy pandas
  python seg_pipeline.py prep    --root FloodNet_official --vqa vqa_full.csv   (cache, long side 2048)
  python seg_pipeline.py train   --fold 0 --out runs/seg_f0            (use CUDA_VISIBLE_DEVICES=<gpu>)
  python seg_pipeline.py predict --fold 0 --out runs/seg_f0            (writes masks into --pred_dir)
  python seg_pipeline.py score                                          (mIoU over everything in --pred_dir)
<root> contains *.jpg images and *_lab.png masks (official class-id PNGs, or colour masks). Use the official
FloodNet-Supervised_v1.0 download: the old Images/ copy is truncated.
Leakage rule: labelled VQA images keep the 5 image-level CV folds of vqa_full.csv. The fold-k segmenter trains on
all masked images WITHOUT VQA answers (minus a fixed seg_val set) plus the VQA images of the other 4 folds, and
predicts fold k only, so every VQA image gets a mask from a model that never saw it.
--fold -1 trains on every masked image except seg_val (final model for the other datasets)."""
import os, glob, json, random, argparse, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image, ImageOps
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file

from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False  # fail loudly: the old Images/ copy was cut at 2.25/2.5 MiB per file and
                                         # silently decoded as ~50% grey; prep now SKIPs such files
CLASSES = ["background", "building_flooded", "building_nonflooded", "road_flooded", "road_nonflooded",
           "water", "tree", "vehicle", "pool", "grass"]
PAL = np.array([(0, 0, 0), (255, 0, 0), (180, 120, 120), (160, 150, 20), (140, 140, 140),
                (61, 230, 250), (0, 82, 255), (255, 0, 245), (255, 235, 0), (4, 250, 7)], dtype=np.int32)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

def unpin(*_):
    """Forked workers inherit a one-core CPU mask on this server ({0,24}); all prep/DataLoader workers then share
    one core. Give each worker every CPU back."""
    os.sched_setaffinity(0, range(os.cpu_count()))

# ---------------------------------------------------------------- prep
def _prep_one(a):
    try:
        return _prep_core(a), None
    except Exception as e:
        return a[0], f"{type(e).__name__}: {e}"

def _prep_core(a):
    iid, ip, mp, L, out = a
    if os.path.exists(f"{out}/img/{iid}.jpg") and os.path.exists(f"{out}/lab/{iid}.png"): return iid
    im, m = Image.open(ip), Image.open(mp)   # masks may be smaller (local copy: 1024x1024) but span the full image
    im.draft("RGB", (L, L))
    im = ImageOps.exif_transpose(im)         # 57 photos carry EXIF orientation 3 (180 deg); labels follow the rotated view
    s = L / max(im.size); size = (round(im.width * s), round(im.height * s))
    im.convert("RGB").resize(size, Image.BILINEAR).save(f"{out}/img/{iid}.jpg", quality=95)
    if m.mode in ("L", "P", "I"):                                  # official FloodNet: class ids 0-9
        lab = np.asarray(m).astype(np.uint8); lab = np.where(lab <= 9, lab, 255).astype(np.uint8)
    else:                                                          # colour masks: match at mask resolution
        arr = np.asarray(m.convert("RGB")).astype(np.int32)
        d = np.abs(arr[:, :, None, :] - PAL[None, None]).sum(3)
        lab = np.where(d.min(2) <= 30, d.argmin(2), 255).astype(np.uint8)   # 255 = ignore (unmatched colour)
    Image.fromarray(lab).resize(size, Image.NEAREST).save(f"{out}/lab/{iid}.png")
    return iid

def prep(a):
    os.makedirs(f"{a.cache}/img", exist_ok=True); os.makedirs(f"{a.cache}/lab", exist_ok=True)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    imgs, masks = {}, {}
    skip = {os.path.basename(os.path.normpath(a.cache)), "runs", "seg_cache"}
    for d, dirs, files in os.walk(a.root):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in files:
            p, fl = os.path.join(d, f), f.lower()
            if fl.endswith("_lab.png"): masks[f[:-8]] = p
            elif fl.endswith((".jpg", ".jpeg")): imgs[stem(p)] = p
    ids = sorted(set(imgs) & set(masks))
    print(f"root: {os.path.abspath(a.root)}\nimages {len(imgs)}  masks {len(masks)}  both {len(ids)}")
    if not ids:
        raise SystemExit("No image/mask pairs found. Point --root at the folder that contains the .JPG images "
                         "and the *_lab.png masks (run: ls <root> and find <root> -name '*_lab.png' | head).")
    with Pool(a.workers, initializer=unpin) as p:
        res = p.map(_prep_one, [(i, imgs[i], masks[i], a.L, a.cache) for i in ids], chunksize=8)
    for i, e in res:
        if e: print("SKIPPED", i, e)
    ids = [i for i, e in res if not e]
    v = pd.read_csv(a.vqa); v = v[v.answer.notna()]
    fold = dict(zip(v.Image_ID.str.rsplit(".", n=1).str[0], v.fold.astype(int)))
    vqa = {i: fold[i] for i in ids if i in fold}
    rest = [i for i in ids if i not in vqa]
    random.Random(0).shuffle(rest)
    nv = max(1, len(rest) // 10)
    json.dump({"seg_val": rest[:nv], "seg_train": rest[nv:], "vqa": vqa}, open(f"{a.cache}/split.json", "w"))
    print(f"seg_train {len(rest) - nv}  seg_val {nv}  vqa {len(vqa)} (folds {np.bincount(list(vqa.values()))})")

def fold_ids(sp, k):
    """(train, predict) ids for fold k; k = -1 -> train on all VQA images, predict none."""
    return (sp["seg_train"] + [i for i, f in sp["vqa"].items() if f != k],
            [i for i, f in sp["vqa"].items() if f == k])

# ---------------------------------------------------------------- data / eval
class DS(Dataset):
    """crop > 0: random-scale square crop (scale 0.5-1.5 of the cached size), rot90/flip, brightness/contrast.
    crop = 0: the whole cached image (use batch size 1, sizes differ)."""
    def __init__(s, ids, root, crop=0): s.ids, s.root, s.crop = ids, root, crop
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        im = np.asarray(Image.open(f"{s.root}/img/{s.ids[i]}.jpg").convert("RGB"))
        lab = np.asarray(Image.open(f"{s.root}/lab/{s.ids[i]}.png"))
        if s.crop:
            cs = int(s.crop / random.uniform(.5, 1.5))                       # window side in cached pixels
            ph, pw = max(cs - im.shape[0], 0), max(cs - im.shape[1], 0)
            if ph or pw:
                im = np.pad(im, ((0, ph), (0, pw), (0, 0))); lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=255)
            y, x = random.randint(0, im.shape[0] - cs), random.randint(0, im.shape[1] - cs)
            im = np.asarray(Image.fromarray(im[y:y + cs, x:x + cs]).resize((s.crop, s.crop), Image.BILINEAR))
            lab = np.asarray(Image.fromarray(lab[y:y + cs, x:x + cs]).resize((s.crop, s.crop), Image.NEAREST))
            k = random.randrange(4); im, lab = np.rot90(im, k), np.rot90(lab, k)
            if random.random() < .5: im, lab = im[:, ::-1], lab[:, ::-1]
            im = im.astype(np.float32); mu = im.mean()
            im = np.clip((im - mu) * random.uniform(.8, 1.2) + mu * random.uniform(.8, 1.2), 0, 255).astype(np.uint8)
        # uint8 out, normalised on the GPU (to_dev): float32/int64 batches made worker->main transfer the bottleneck
        return (torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(lab)), s.ids[i])

def to_dev(x, y, dev):
    m, sd = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in (MEAN, STD))
    return (x.to(dev, non_blocking=True).float() / 255 - m) / sd, y.to(dev, non_blocking=True).long()

def up(lg, size): return F.interpolate(lg.float(), size=size, mode="bilinear", align_corners=False)

def amp_dtype(a): return torch.bfloat16 if a.amp == "bf16" else torch.float16

@torch.no_grad()
def infer(model, x, dt, crop=1024, stride=768, tta=False):
    """Class probabilities for a (1,3,H,W) image from overlapping crop x crop windows (the training scale).
    Whole-image inference at 2048x1536 OOMs on 11 GB cards: the decode head upsamples 4x768 channels in fp32."""
    _, _, H, W = x.shape
    starts = lambda n: sorted({*range(0, max(n - crop, 0) + 1, stride), max(n - crop, 0)})
    pr = torch.zeros(1, 10, H, W, device=x.device); cnt = torch.zeros(1, 1, H, W, device=x.device)
    for y0 in starts(H):
        for x0 in starts(W):
            w = x[..., y0:y0 + crop, x0:x0 + crop]
            with torch.autocast("cuda", dtype=dt):
                p = up(model(pixel_values=w).logits, w.shape[-2:]).softmax(1)
                if tta: p += up(model(pixel_values=w.flip(-1)).logits, w.shape[-2:]).softmax(1).flip(-1)
            pr[..., y0:y0 + crop, x0:x0 + crop] += p; cnt[..., y0:y0 + crop, x0:x0 + crop] += 1
    return pr / cnt

@torch.no_grad()
def run_eval(model, loader, dev, dt, save_dir=None, tta=False):
    """loader must yield single whole images (batch size 1)."""
    model.eval(); conf = torch.zeros(10, 10, dtype=torch.long, device=dev)
    for x, y, ids in loader:
        x, y = to_dev(x, y, dev)
        p = infer(model, x, dt, tta=tta).argmax(1)
        k = y != 255
        conf += torch.bincount(y[k] * 10 + p[k], minlength=100).view(10, 10)
        if save_dir:
            for pi, iid in zip(p.cpu().numpy().astype(np.uint8), ids): Image.fromarray(pi).save(f"{save_dir}/{iid}.png")
    return conf

def iou(conf):
    tp = conf.diag().float(); union = conf.sum(0) + conf.sum(1) - tp
    return torch.where(union > 0, tp / union.clamp(min=1), torch.full_like(tp, float("nan"))).cpu().numpy()

def show(v, tag, log=print):
    log(f"{tag}  mIoU {np.nanmean(v) * 100:.2f}   " +
        "  ".join(f"{c[:14]} {x * 100:.1f}" for c, x in zip(CLASSES, v)))

def _sdpa_attention(self, hidden_states, height, width, output_attentions=False):
    """transformers' SegformerEfficientSelfAttention.forward, but via F.scaled_dot_product_attention: the eager
    version materialises the full attention matrix (softmax in fp32), so 1024px crops do not fit in 24 GB."""
    b, n, _ = hidden_states.shape
    heads = lambda t: t.view(b, t.shape[1], -1, self.attention_head_size).transpose(1, 2)
    q = heads(self.query(hidden_states))
    if self.sr_ratio > 1:
        hidden_states = self.sr(hidden_states.transpose(1, 2).reshape(b, -1, height, width))
        hidden_states = self.layer_norm(hidden_states.flatten(2).transpose(1, 2))
    k, v = heads(self.key(hidden_states)), heads(self.value(hidden_states))
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
    return (o.transpose(1, 2).reshape(b, n, self.all_head_size),)

def build_model(name, revision, pretrained=True):
    """pretrained: HF weights (safetensors revision); else architecture only, weights come from load_state_dict.
    Checkpoints are saved as plain state_dicts: transformers 5.x save_pretrained renames decode_head.linear_c ->
    linear_projections and does not rename back on load, which silently re-initialised the decoder before."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    from transformers.models.segformer import modeling_segformer
    modeling_segformer.SegformerEfficientSelfAttention.forward = _sdpa_attention
    kw = dict(num_labels=10, id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)})
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(name, revision=revision, ignore_mismatched_sizes=True, **kw)
    return SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(name, revision=revision, **kw))

def load_trained(out, dev):
    cfg = json.load(open(f"{out}/args.json"))
    model = build_model(cfg["model"], cfg["revision"], pretrained=False)
    model.load_state_dict(load_file(f"{out}/best.safetensors"))   # strict: any key mismatch raises
    return model.to(dev)

# ---------------------------------------------------------------- train / predict / score
def train(a):
    dev, dt = "cuda", amp_dtype(a)
    sp = json.load(open(f"{a.cache}/split.json")); tr, _ = fold_ids(sp, a.fold)
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    logf = open(f"{a.out}/log.txt", "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"fold {a.fold}: train {len(tr)} images, seg_val {len(sp['seg_val'])}, args {vars(a)}")
    cnt = np.zeros(10)
    for i in tr: cnt += np.bincount(np.asarray(Image.open(f"{a.cache}/lab/{i}.png")).ravel(), minlength=256)[:10]
    w = 1 / np.sqrt(cnt / cnt.sum() + 1e-6); w = np.clip(w / w.mean(), 0.2, 5)
    log(f"class weights: {dict(zip(CLASSES, w.round(2)))}")
    w = torch.tensor(w, dtype=torch.float32, device=dev)
    tl = DataLoader(DS(tr, a.cache, a.crop), a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    worker_init_fn=unpin, persistent_workers=True)
    vl = DataLoader(DS(sp["seg_val"], a.cache), 1, num_workers=4, worker_init_fn=unpin)
    model = build_model(a.model, a.revision).to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1, (it + 1) / a.warmup) * (1 - it / a.iters) ** 1.0)
    scaler = torch.amp.GradScaler(enabled=a.amp == "fp16")
    best, it, micro, tot, t0 = -1, 0, 0, 0.0, time.time()
    while it < a.iters:
        model.train()
        for x, y, _ in tl:
            x, y = to_dev(x, y, dev)
            with torch.autocast("cuda", dtype=dt):
                lg = model(pixel_values=x).logits
            loss = F.cross_entropy(up(lg, y.shape[-2:]), y, weight=w, ignore_index=255) / a.accum
            scaler.scale(loss).backward(); tot += loss.item(); micro += 1
            if micro % a.accum: continue                          # optimizer step every --accum batches
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); it += 1
            if it % 100 == 0:
                log(f"iter {it}/{a.iters} loss {tot / 100:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / 100:.2f}s/it"); tot, t0 = 0.0, time.time()
            if it % a.eval_every == 0 or it == a.iters:
                v = iou(run_eval(model, vl, dev, dt)); show(v, f"iter {it} seg_val:", log)
                if np.nanmean(v) > best:
                    best = np.nanmean(v); save_file({k: t.contiguous() for k, t in model.state_dict().items()},
                                                    f"{a.out}/best.safetensors"); log("  saved best")
                model.train(); t0 = time.time()
            if it >= a.iters: break
    log(f"done, best seg_val mIoU {best * 100:.2f}")

def predict(a):
    dev = "cuda"; sp = json.load(open(f"{a.cache}/split.json")); _, te = fold_ids(sp, a.fold)
    if not te: raise SystemExit(f"fold {a.fold} has no images to predict")
    model = load_trained(a.out, dev); os.makedirs(a.pred_dir, exist_ok=True)
    conf = run_eval(model, DataLoader(DS(te, a.cache), 1, num_workers=4, worker_init_fn=unpin), dev, amp_dtype(a), a.pred_dir, a.tta)
    show(iou(conf), f"fold {a.fold}: {len(te)} VQA images (never seen by this segmenter):")

def score(a):
    ids = [os.path.basename(p)[:-4] for p in glob.glob(f"{a.pred_dir}/*.png")]
    conf = torch.zeros(10, 10, dtype=torch.long)
    for i in ids:
        y = torch.from_numpy(np.asarray(Image.open(f"{a.cache}/lab/{i}.png")).astype(np.int64))
        p = torch.from_numpy(np.asarray(Image.open(f"{a.pred_dir}/{i}.png")).astype(np.int64))
        k = y != 255; conf += torch.bincount(y[k] * 10 + p[k], minlength=100).view(10, 10)
    show(iou(conf), f"{a.pred_dir}: {len(ids)} images")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["prep", "train", "predict", "score"])
    ap.add_argument("--root", default="."); ap.add_argument("--vqa", default="vqa_full.csv")
    ap.add_argument("--cache", default="seg_cache_official"); ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--out", default="runs/seg_f0")
    ap.add_argument("--pred_dir", default="runs/seg_oof/preds"); ap.add_argument("--tta", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--revision", default="refs/pr/2", help="hub revision holding model.safetensors")
    ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16", help="fp16 for pre-Ampere GPUs (2080 Ti)")
    a = ap.parse_args()
    {"prep": prep, "train": train, "predict": predict, "score": score}[a.stage](a)
