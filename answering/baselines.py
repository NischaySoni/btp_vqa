"""Usage: python answering/baselines.py
Blind baselines (no image), the lower bound in the sense of Prompt-RSVQA's "visual blind" model:
- prior:          most frequent answer per intent in the dataset's own training data
                  (FloodNet: the other 4 CV folds; AIFloodSense: train split; LADI: validation split), scored on held-out rows;
- floodnet_prior: most frequent FloodNet answer per intent applied to another dataset (transfer without looking);
                  intents FloodNet never asks fall back to the dataset prior.
Writes answering/out/<dataset>/<method>.jsonl and prints scores per question type."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_qa, score, show, ROOT

TRAIN_SPLIT = {"aifloodsense": "train", "ladi_v2": "validation"}

def majority(rows):
    c = collections.defaultdict(collections.Counter)
    for r in rows: c[r["intent"]][r["answer"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in c.items()}

def write(dataset, name, preds):
    os.makedirs(f"{ROOT}/answering/out/{dataset}", exist_ok=True)
    with open(f"{ROOT}/answering/out/{dataset}/{name}.jsonl", "w") as f:
        for q, p in sorted(preds.items()): f.write(json.dumps(dict(qid=q, pred=p)) + "\n")

def main():
    fn = load_qa("floodnet"); preds = {}
    for k in range(5):
        prior = majority([r for r in fn if r["meta"]["fold"] != k])
        preds.update({i: prior[r["intent"]] for i, r in enumerate(fn) if r["meta"]["fold"] == k})
    write("floodnet", "prior", preds); show(score(fn, preds), "floodnet / prior (5-fold CV)")
    fn_prior = majority(fn)
    for ds, tr in TRAIN_SPLIT.items():
        qa = load_qa(ds); test = {i for i, r in enumerate(qa) if r["split"] != tr}
        prior = majority([r for r in qa if r["split"] == tr])
        for name, pr in (("prior", prior), ("floodnet_prior", {**prior, **fn_prior})):
            preds = {i: pr.get(qa[i]["intent"]) for i in test}
            write(ds, name, preds)
            show(score([qa[i] for i in sorted(test)], {j: preds[i] for j, i in enumerate(sorted(test))}),
                 f"{ds} / {name} (test rows only)")

if __name__ == "__main__":
    main()
