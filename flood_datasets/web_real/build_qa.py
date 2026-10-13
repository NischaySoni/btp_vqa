"""Usage: python flood_datasets/web_real/build_qa.py [--annotators a,b]   (default: every annotations_*.csv)
Turns annotation sheets from annotate.py into qa.jsonl (shared schema) for our real web flood set.
One annotator: every kept image is used (answer_source "human").
Two or more: prints per-field agreement (Cohen's kappa for categories; exact / within-tolerance for counts) and keeps
an answer only where annotators agree; counts agree if |a - b| <= max(1, 10% of the mean), and the rounded mean is used.
Disagreements are written to disagreements.csv for adjudication (fill it in as another annotations_<name>.csv)."""
import os, sys, csv, glob, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import row, write_jsonl, summary

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = {"environment": "environment", "sky": "sky_yn", "overall_condition": "overall_condition",
       "road_condition": "road_condition"}
CNT = {"count_total": "count_total", "count_flooded": "count_flooded"}

def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def as_int(v): return int(v) if str(v).strip().isdigit() else None

def merge(vals, field):
    vals = [v for v in vals if v not in ("", None)]
    if not vals: return None
    if field in CNT:
        nums = [as_int(v) for v in vals]
        if any(x is None for x in nums): return None              # "many" / not countable -> no counting question
        m = sum(nums) / len(nums)
        return round(m) if max(nums) - min(nums) <= max(1, 0.1 * m) else "DISAGREE"
    return vals[0] if len(set(vals)) == 1 else "DISAGREE"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--annotators", default=""); a = ap.parse_args()
    files = ([f"{HERE}/annotations_{x}.csv" for x in a.annotators.split(",")] if a.annotators
             else sorted(glob.glob(f"{HERE}/annotations_*.csv")))
    ann = {os.path.basename(f)[12:-4]: {r["id"]: r for r in csv.DictReader(open(f))} for f in files}
    meta = {r["id"]: r for r in csv.DictReader(open(f"{HERE}/candidates.csv"))}
    who = list(ann); ids = sorted(set.intersection(*(set(d) for d in ann.values())))
    print(f"annotators {who}; images annotated by all: {len(ids)}")
    if len(who) > 1:
        for f in ["keep"] + list(CAT):
            pairs = [(ann[who[0]][i][f], ann[who[1]][i][f]) for i in ids if ann[who[0]][i][f] and ann[who[1]][i][f]]
            if pairs: print(f"  {f:18s} n={len(pairs):4d} agree={sum(x == y for x, y in pairs) / len(pairs):.2f} kappa={kappa(pairs):.2f}")
        for f in CNT:
            pairs = [(as_int(ann[who[0]][i][f]), as_int(ann[who[1]][i][f])) for i in ids]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs: print(f"  {f:18s} n={len(pairs):4d} exact={sum(x == y for x, y in pairs) / len(pairs):.2f} "
                            f"within1={sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs):.2f}")
    rows, dis = [], []
    for i in ids:
        if merge([ann[w][i]["keep"] for w in who], "keep") != "yes": continue
        m = meta[i]; img = f"{HERE}/images/{i}.jpg"
        info = dict(viewpoint=merge([ann[w][i]["viewpoint"] for w in who], "viewpoint"), source_url=m["page"],
                    licence=m["licence"], author=m["artist"], found_via=m["found_via"])
        for f, intent in {**CAT, **CNT}.items():
            v = merge([ann[w][i][f] for w in who], f)
            if v == "DISAGREE": dis.append(dict(id=i, field=f, **{w: ann[w][i][f] for w in who})); continue
            if v is None or v == "no road": continue
            rows.append(row("web_real", "test", i, img, intent, v, "human", **info))
        tot, fl = (merge([ann[w][i][f] for w in who], f) for f in ("count_total", "count_flooded"))
        if isinstance(tot, int) and isinstance(fl, int) and fl <= tot:
            rows.append(row("web_real", "test", i, img, "count_nonflooded", tot - fl, "derived", **info))
            rows.append(row("web_real", "test", i, img, "building_flooded_yn", "yes" if fl else "no", "derived", **info))
    write_jsonl(rows, f"{HERE}/qa.jsonl"); summary(rows)
    if dis:
        with open(f"{HERE}/disagreements.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "field"] + who); w.writeheader(); w.writerows(dis)
        print(f"{len(dis)} disagreements -> disagreements.csv")

if __name__ == "__main__":
    main()
