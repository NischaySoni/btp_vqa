"""Usage: python answering/run_masks.py <floodnet|aifloodsense|ladi_v2> [--labels gt|<pred_dir>] [--fit own|floodnet]
Answers every question from a label map: scene.py builds the scene description, reason.py answers with parameters
fitted on training data only.
--labels gt: ground-truth labels (FloodNet: seg_cache_official/lab, 2048 px, EXIF-corrected; AIFloodSense: its masks),
  i.e. the oracle upper bound of this stage. Otherwise a directory of predicted FloodNet-class PNGs, one <image_id>.png
  per image (FloodNet: runs/seg_oof/preds from the fold models; others: answering/out/<ds>/preds_seg_all).
--fit own: fit on the dataset's own training rows (FloodNet: the 4 other CV folds; AIFloodSense: train split;
  LADI: validation split), score the held-out rows ("calibrated").
--fit floodnet: fit on FloodNet GT scenes only and apply unchanged ("zero-shot transfer"); intents FloodNet never asks
  borrow the parameters of their FloodNet counterpart (MAP), or fall back to own fitting if there is none.
Intents the scene cannot express (e.g. sky with FloodNet classes, damage) fall back to the training-majority answer and
are reported separately. Rows answered from the dataset's own masks (answer_source "mask") are skipped (circular).
Writes answering/out/<dataset>/masks_<labels tag>_<fit>.jsonl."""
import os, sys, json, pickle, argparse, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_qa, score, show, ROOT
import scene as S, reason as R

TRAIN = {"aifloodsense": "train", "ladi_v2": "validation"}
MAP = {"image_flooded_yn": "overall_condition", "building_flooded_yn": "count_flooded"}   # FloodNet counterpart
CLASS_FEATURES = {"floodnet": None, "aif": {"sky_yn", "environment", "building_any_yn", "image_flooded_yn",
                                            "overall_condition", "building_flooded_yn", "count_total", "count_flooded",
                                            "count_nonflooded"}}
FLOODNET_CLASSES_CANNOT = {"sky_yn", "building_damage_yn", "road_damage_yn"}

def _init(): os.sched_setaffinity(0, range(os.cpu_count()))   # forked workers here inherit a 2-core mask

def _scene(args):
    path, kind = args
    return S.load(path, kind) if os.path.exists(path) else None

def scenes_for(dataset, qa, labels, workers):
    ids = sorted({r["image_id"] for r in qa})
    if labels == "gt" and dataset == "floodnet":
        paths, kind = {i: f"{ROOT}/seg_cache_official/lab/{i}.png" for i in ids}, "floodnet"
    elif labels == "gt" and dataset == "aifloodsense":
        paths = {r["image_id"]: f"{ROOT}/{r['image']}".replace("/images/", "/masks/").replace(".jpg", ".png") for r in qa}
        kind = "aif"
    elif labels == "gt":
        raise SystemExit(f"{dataset} has no ground-truth masks; pass a prediction directory")
    else:
        d = labels if os.path.isabs(labels) else f"{ROOT}/{labels}"; paths, kind = {i: f"{d}/{i}.png" for i in ids}, "floodnet"
    tag = "gt" if labels == "gt" else os.path.basename(os.path.normpath(labels))
    cache = f"{ROOT}/answering/out/{dataset}/scenes_{tag}.pkl"
    if os.path.exists(cache):
        sc = pickle.load(open(cache, "rb"))
        if all(i in sc for i in ids): return sc, tag, kind
    with Pool(workers, initializer=_init) as p:
        sc = dict(zip(ids, p.map(_scene, [(paths[i], kind) for i in ids], chunksize=8)))
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(sc, open(cache, "wb"))
    return sc, tag, kind

def can_answer(intent, kind):
    if not R.supported(intent): return False
    return intent not in FLOODNET_CLASSES_CANNOT if kind == "floodnet" else intent in CLASS_FEATURES["aif"]

def fit_rows(qa, sc, rows, intent):
    rows = [i for i in rows if qa[i]["intent"] == intent and sc.get(qa[i]["image_id"]) is not None]
    return R.fit([sc[qa[i]["image_id"]] for i in rows], [qa[i]["answer"] for i in rows], intent) if rows else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["floodnet", "aifloodsense", "ladi_v2"])
    ap.add_argument("--labels", default="gt"); ap.add_argument("--fit", choices=["own", "floodnet"], default="own")
    ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    qa = load_qa(a.dataset); sc, tag, kind = scenes_for(a.dataset, qa, a.labels, a.workers)
    use = [i for i, r in enumerate(qa) if r["answer_source"] != "mask"]
    if a.dataset == "floodnet":
        groups = [([i for i in use if qa[i]["meta"]["fold"] != k], [i for i in use if qa[i]["meta"]["fold"] == k])
                  for k in range(5)]
    else:
        groups = [([i for i in use if qa[i]["split"] == TRAIN[a.dataset]], [i for i in use if qa[i]["split"] != TRAIN[a.dataset]])]
    if a.fit == "floodnet":
        fqa = load_qa("floodnet"); fsc, _, _ = scenes_for("floodnet", fqa, "gt", a.workers); fall = list(range(len(fqa)))
    preds, how, params = {}, {}, []
    for tr, te in groups:
        prior = {}
        for i in tr: prior.setdefault(qa[i]["intent"], collections.Counter())[qa[i]["answer"]] += 1
        p = {}
        for it in sorted({qa[i]["intent"] for i in te}):
            if not can_answer(it, kind): p[it] = None; continue
            if a.fit == "floodnet" and (it in R.COUNTS or it in R.FEATURES and it in {r["intent"] for r in fqa} or it in MAP):
                p[it] = fit_rows(fqa, fsc, fall, MAP.get(it, it))
            else:
                p[it] = fit_rows(qa, sc, tr, it)
        params.append(p)
        for i in te:
            it, s = qa[i]["intent"], sc.get(qa[i]["image_id"])
            if p.get(it) is None or s is None:
                preds[i] = prior[it].most_common(1)[0][0] if it in prior else None; how[i] = "prior"
            else:
                v = R.answer(s, it, p[it])
                if it == "image_flooded_yn" and v in ("flooded", "non flooded"): v = "yes" if v == "flooded" else "no"
                preds[i] = v; how[i] = "scene"
    out = f"{ROOT}/answering/out/{a.dataset}/masks_{tag}_{a.fit}.jsonl"
    with open(out, "w") as f:
        for q, v in sorted(preds.items()): f.write(json.dumps(dict(qid=q, pred=v, how=how[q])) + "\n")
    scored = sorted(preds); sub = [qa[i] for i in scored]; sp = {j: preds[i] for j, i in enumerate(scored)}
    show(score(sub, sp, "question_type"), f"{a.dataset} / labels={tag} fit={a.fit}  ({len(scored)} rows scored)")
    show(score(sub, sp, "intent"), "  by intent:")
    fb = collections.Counter(qa[i]["intent"] for i in scored if how[i] == "prior")
    if fb: print("  answered by blind prior (scene cannot express them):", dict(fb))
    print("  fitted parameters:", json.dumps(params, default=float)[:500])

if __name__ == "__main__":
    main()
