"""Usage: python answering/evaluate.py <dataset> <preds.jsonl> [--by intent|question_type|answer_source|fold]
Scores predictions against flood_datasets/<dataset>/qa.jsonl. preds.jsonl: one {"qid": <row number in qa.jsonl>,
"pred": <answer>} per line; rows without a prediction count as wrong. Answers are compared after common.norm (counts as
ints). Reports accuracy per group plus, for counting intents, exact / within-1 / MAE. FloodNet rows carry meta.fold,
so --by fold gives the per-fold spread of the 5-fold CV."""
import os, sys, json, argparse, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/flood_datasets")
from common import norm

COUNT = {"count_total", "count_flooded", "count_nonflooded"}

def load_qa(dataset):
    return [json.loads(l) for l in open(f"{ROOT}/flood_datasets/{dataset}/qa.jsonl")]

def score(qa, preds, by="question_type"):
    """preds: {qid: answer}. Returns {group: dict(n, acc, [within1, mae])} incl. 'ALL'."""
    g = collections.defaultdict(list)
    for i, r in enumerate(qa):
        p = norm(preds.get(i)); key = r["meta"].get("fold") if by == "fold" else r[by]
        for k in (key, "ALL"): g[k].append((r, p))
    out = {}
    for k, items in sorted(g.items(), key=lambda x: str(x[0])):
        d = dict(n=len(items), acc=sum(p == r["answer"] for r, p in items) / len(items))
        cnt = [(r["answer"], p) for r, p in items if r["intent"] in COUNT]
        if cnt:
            err = [abs(a - p) if isinstance(p, int) else None for a, p in cnt]
            d["within1"] = sum(e is not None and e <= 1 for e in err) / len(cnt)
            d["mae"] = sum(e if e is not None else a for e, (a, _) in zip(err, cnt)) / len(cnt)  # no answer -> |a - 0|
        out[k] = d
    return out

def show(res, title=""):
    if title: print(title)
    for k, d in res.items():
        extra = f"  within1 {d['within1'] * 100:5.1f}  MAE {d['mae']:5.2f}" if "mae" in d else ""
        print(f"  {str(k):24s} n={d['n']:5d}  acc {d['acc'] * 100:5.1f}{extra}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("preds")
    ap.add_argument("--by", default="question_type"); a = ap.parse_args()
    preds = {}
    for l in open(a.preds):
        d = json.loads(l); preds[d["qid"]] = d["pred"]
    show(score(load_qa(a.dataset), preds, a.by), f"{a.dataset}: {a.preds}")

if __name__ == "__main__":
    main()
