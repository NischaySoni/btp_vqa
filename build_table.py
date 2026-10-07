"""Usage: python build_full_table.py <root> [out.csv]
Expects <root>/Questions/*.json  ({idx: {Image_ID, Question, Question_Type, Ground_Truth?}})
and <root>/Images/**.jpg. Writes one row per image-question pair with:
split (from file name), canonical intent, normalised answer, image path, 5 image-level CV folds."""
import sys, os, json, glob, random, collections
import pandas as pd

def intent(q):
    q = q.lower().replace("-", " ")
    if "overall condition" in q: return "overall_condition"
    if "entire road" in q: return "road_yn_nonflooded" if "non flooded" in q else "road_yn_flooded"
    if "condition of" in q and "road" in q: return "road_condition"
    if "how many" in q:
        if "non flooded" in q: return "count_nonflooded"
        if "flooded" in q: return "count_flooded"
        return "count_total"
    return "other"

def norm(a):
    if a is None or (isinstance(a, float) and pd.isna(a)): return None
    s = str(a).strip().lower().replace("-", " ")
    return int(s) if s.isdigit() else s

def main(root, out="vqa_full.csv"):
    img_path = {}
    for p in glob.glob(os.path.join(root, "Images", "**", "*.*"), recursive=True):
        if p.lower().endswith((".jpg", ".jpeg", ".png")): img_path[os.path.basename(p)] = p
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "Questions", "*.json"))):
        split = os.path.basename(f).split("_")[0].lower()
        data = json.load(open(f, encoding="utf-8"))
        items = data.values() if isinstance(data, dict) else data
        for d in items:
            r = dict(d); r["split"] = split
            r["intent"] = intent(d["Question"])
            r["answer"] = norm(d.get("Ground_Truth", d.get("Answer")))
            r["path"] = img_path.get(d["Image_ID"])
            rows.append(r)
    df = pd.DataFrame(rows)
    # 5 image-level folds over labelled (train) images, fixed seed
    imgs = sorted(df[df.answer.notna()].Image_ID.unique())
    random.Random(0).shuffle(imgs)
    fold = {im: i % 5 for i, im in enumerate(imgs)}
    df["fold"] = df.Image_ID.map(fold)
    df.to_csv(out, index=False)

    print(f"rows: {len(df)} -> {out}\ncolumns: {list(df.columns)}")
    print("\nrows / images / answered per split:")
    print(df.groupby("split").agg(rows=("Question", "size"), images=("Image_ID", "nunique"),
                                  answered=("answer", lambda s: s.notna().sum())))
    print("\nmissing image files:", int(df.path.isna().sum()))
    print("\nlabelled rows per intent:")
    print(df[df.answer.notna()].groupby("intent").size())
    print("\nanswer distribution per intent (labelled):")
    for k, g in df[df.answer.notna()].groupby("intent"):
        print(f"  {k}: {collections.Counter(g.answer).most_common(8)}")
    print("\nunique question texts:", df.Question.nunique())
    print("images with no overall_condition question:",
          len(set(df.Image_ID) - set(df[df.intent == 'overall_condition'].Image_ID)))

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vqa_full.csv")