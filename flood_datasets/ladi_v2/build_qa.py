"""Usage: python flood_datasets/ladi_v2/build_qa.py [--seed 0]
Needs the LADI v2 validation + test splits (v2a labels) in flood_datasets/ladi_v2/raw/data/{validation,test}/*.arrow:
  huggingface_hub.snapshot_download('MITLL/LADI-v2-dataset', repo_type='dataset', local_dir='flood_datasets/ladi_v2/raw',
                                    allow_patterns=['README.md', 'data/dataset_dict.json', 'data/test/*', 'data/validation/*'])
Selects every flooding_any image plus an equal number of non-flooded ones (half "hard" negatives that show ordinary
water, half with buildings and no water), extracts them (original bytes, PNG) to images/<split>_<n>.png, and writes qa.jsonl with yes/no and
condition questions answered from the FEMA-trained annotators' image labels (answer_source "label").
LADI has no counts, so there are no counting questions (a hand-counted subset is planned)."""
import os, sys, glob, random, argparse
import pandas as pd, pyarrow as pa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import row, write_jsonl, summary

HERE = os.path.dirname(os.path.abspath(__file__))
YN = {"image_flooded_yn": "flooding_any", "building_flooded_yn": "flooding_structures", "building_any_yn": "buildings_any",
      "building_damage_yn": "buildings_affected_or_greater", "road_any_yn": "roads_any", "road_damage_yn": "roads_damage",
      "water_any_yn": "water_any"}

def load(split):
    parts = []
    for f in sorted(glob.glob(f"{HERE}/raw/data/{split}/*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f)).read_all().to_pandas()
        t["image_bytes"] = t.image.map(lambda d: d["bytes"]); parts.append(t.drop(columns="image"))
    df = pd.concat(parts, ignore_index=True); df["n"] = range(len(df)); return df

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    os.makedirs(f"{HERE}/images", exist_ok=True); rows = []
    for split in ("validation", "test"):
        df = load(split); rng = random.Random(a.seed)
        pos = df[df.flooding_any]
        hard = df[~df.flooding_any & df.water_any]; plain = df[~df.flooding_any & ~df.water_any & df.buildings_any]
        nh = min(len(hard), len(pos) // 2); neg = (hard.loc[rng.sample(list(hard.index), nh)].assign(neg="hard_water"),
                                                     plain.loc[rng.sample(list(plain.index), min(len(plain), len(pos) - nh))]
                                                     .assign(neg="buildings_no_water"))
        sel = pd.concat([pos.assign(neg=""), *neg])
        print(f"{split}: {len(pos)} flooded + {nh} hard negatives (water, no flood) + {len(neg[1])} other negatives")
        for _, r in sel.iterrows():
            ext = "png" if r.image_bytes[:4] == b"\x89PNG" else "jpg"
            iid = f"{split}_{r.n:04d}"; img = f"{HERE}/images/{iid}.{ext}"
            if not os.path.exists(img): open(img, "wb").write(r.image_bytes)
            info = dict(selection=r.neg or "flooded")
            for intent, col in YN.items():
                rows.append(row("ladi_v2", split, iid, img, intent, "yes" if r[col] else "no", "label", **info))
            rows.append(row("ladi_v2", split, iid, img, "overall_condition",
                            "flooded" if r.flooding_any else "non flooded", "label", **info))
    write_jsonl(rows, f"{HERE}/qa.jsonl"); summary(rows)

if __name__ == "__main__":
    main()
