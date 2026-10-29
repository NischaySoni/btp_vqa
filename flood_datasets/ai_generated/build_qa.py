"""Build human-verified QA from AI-generated images after annotation.

Usage: python flood_datasets/ai_generated/build_qa.py [--annotators reviewer1,reviewer2]
The review UI hides prompts and intended answers. This builder compares the human consensus with the intended values,
reports agreement and generation fidelity, and writes qa.jsonl plus disagreements.csv.
"""
import os, csv, json, glob, argparse, collections
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import row, write_jsonl, summary

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = ["keep", "viewpoint", "environment", "sky", "overall_condition", "road_condition"]
COUNTS = ["count_total", "count_flooded"]
INTENT = {
    "environment": "environment", "sky": "sky_yn", "overall_condition": "overall_condition",
    "road_condition": "road_condition", "count_total": "count_total", "count_flooded": "count_flooded",
}

def agreement(pairs):
    if not pairs: return None, None
    exact = sum(a == b for a, b in pairs) / len(pairs)
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    expected = sum(ca[k] * cb[k] for k in ca) / len(pairs) ** 2
    kappa = (exact - expected) / (1 - expected) if expected < 1 else 1.0
    return exact, kappa

def as_int(value):
    try: return int(value) if str(value).strip().isdigit() else None
    except (TypeError, ValueError): return None

def merge(values, field):
    values = [v for v in values if v not in ("", None)]
    if not values: return None
    if field in COUNTS:
        nums = [as_int(v) for v in values]
        if any(v is None for v in nums): return None
        mean = sum(nums) / len(nums)
        return round(mean) if max(nums) - min(nums) <= max(1, 0.1 * mean) else "DISAGREE"
    return values[0] if len(set(values)) == 1 else "DISAGREE"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--annotators", default="")
    args = ap.parse_args()
    files = ([f"{HERE}/annotations_{x}.csv" for x in args.annotators.split(",") if x]
             if args.annotators else sorted(glob.glob(f"{HERE}/annotations_*.csv")))
    files = [p for p in files if os.path.isfile(p)]
    if not files: raise SystemExit("No annotation CSVs found yet.")
    annotations = {}
    for path in files:
        name = os.path.basename(path)[12:-4]
        annotations[name] = {r["id"]: r for r in csv.DictReader(open(path, newline=""))}
    names = list(annotations)
    if len(names) < 2: print("WARNING: only one annotator file found; inter-annotator agreement cannot be measured.")
    shared_ids = sorted(set.intersection(*(set(d) for d in annotations.values())))
    specs = {}
    for line in open(f"{HERE}/specs.jsonl"):
        spec = json.loads(line); specs[spec["id"]] = spec
    print(f"Annotators {names}; images annotated by all: {len(shared_ids)}")

    if len(names) >= 2:
        a, b = (annotations[n] for n in names[:2])
        for field in CATEGORIES:
            pairs = [(a[i][field], b[i][field]) for i in shared_ids if a[i].get(field) and b[i].get(field)]
            exact, kappa = agreement(pairs)
            if exact is not None: print(f"{field:20s} n={len(pairs):4d} agreement={exact:.2f} kappa={kappa:.2f}")
        for field in COUNTS:
            pairs = [(as_int(a[i].get(field)), as_int(b[i].get(field))) for i in shared_ids]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs:
                print(f"{field:20s} n={len(pairs):4d} exact={sum(x == y for x,y in pairs)/len(pairs):.2f} "
                      f"within1={sum(abs(x-y) <= 1 for x,y in pairs)/len(pairs):.2f}")

    consensus, disagreements = {}, []
    for iid in shared_ids:
        merged = {}
        for field in CATEGORIES + COUNTS:
            value = merge([annotations[n][iid].get(field, "") for n in names], field)
            merged[field] = value
            if value == "DISAGREE":
                disagreements.append(dict(id=iid, field=field,
                                          **{n: annotations[n][iid].get(field, "") for n in names}))
        consensus[iid] = merged

    for field in ["environment", "sky", "overall_condition", "road_condition", *COUNTS]:
        pairs = [(consensus[i][field], specs[i.rsplit("_s", 1)[0]]["intended"].get(field))
                 for i in shared_ids if consensus[i]["keep"] == "yes" and consensus[i][field] not in (None, "DISAGREE")]
        pairs = [(a, b) for a, b in pairs if b is not None]
        if not pairs: continue
        if field in COUNTS:
            errs = [abs(int(a) - int(b)) for a, b in pairs]
            print(f"intended match {field:20s} n={len(errs):4d} exact={(sum(e==0 for e in errs)/len(errs)):.2f} "
                  f"within1={(sum(e<=1 for e in errs)/len(errs)):.2f}")
        else:
            print(f"intended match {field:20s} n={len(pairs):4d} exact={sum(a==b for a,b in pairs)/len(pairs):.2f}")

    rows = []
    for iid in shared_ids:
        c = consensus[iid]
        if c["keep"] != "yes": continue
        spec = specs[iid.rsplit("_s", 1)[0]]; intended = spec["intended"]
        seed = int(iid.rsplit("_s", 1)[1])
        image = f"{HERE}/images/{iid}.png"
        meta = dict(article=spec["article"], source_url=spec["url"], scene_id=spec["id"], seed=seed,
                    intended=intended, license="CC BY 2.5 (Wikinews)")
        for field, intent in INTENT.items():
            value = c[field]
            if value in (None, "DISAGREE"): continue
            if field == "road_condition" and value == "no road":
                rows.append(row("ai_generated", "test", iid, image, "road_any_yn", "no", "derived", **meta))
                continue
            rows.append(row("ai_generated", "test", iid, image, intent, value, "human", **meta))
        total, flooded = c["count_total"], c["count_flooded"]
        total, flooded = as_int(total), as_int(flooded)
        if total is not None:
            rows.append(row("ai_generated", "test", iid, image, "building_any_yn", "yes" if total else "no", "derived", **meta))
        if total is not None and flooded is not None and flooded <= total:
            rows.append(row("ai_generated", "test", iid, image, "count_nonflooded", total - flooded, "derived", **meta))
            rows.append(row("ai_generated", "test", iid, image, "building_flooded_yn", "yes" if flooded else "no", "derived", **meta))
        if c["overall_condition"] in ("flooded", "non flooded"):
            answer = "yes" if c["overall_condition"] == "flooded" else "no"
            rows.append(row("ai_generated", "test", iid, image, "image_flooded_yn", answer, "derived", **meta))

    write_jsonl(rows, f"{HERE}/qa.jsonl"); summary(rows)
    if disagreements:
        with open(f"{HERE}/disagreements.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "field"] + names); writer.writeheader(); writer.writerows(disagreements)
        print(f"{len(disagreements)} annotation disagreements -> disagreements.csv")

if __name__ == "__main__": main()
