"""Shared QA schema for every non-FloodNet dataset (AIFloodSense, LADI v2, our web set, our AI-generated set).
One JSON object per line in <dataset>/qa.jsonl:
  dataset, split, image_id, image (path relative to the btp_vqa root), question, question_type, intent, answer,
  answer_source ("human" = annotated answer, "label" = derived from image-level labels, "mask" = derived from a
  segmentation mask, "derived" = arithmetic on other human answers), meta (dict, optional).
Intents extend build_table.py's FloodNet intents so one answering pipeline can serve all datasets."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # btp_vqa/

# intent -> (FloodNet-style question type, canonical question text)
INTENTS = {
    # FloodNet intents (same names as build_table.py)
    "count_total":        ("Simple_Counting", "How many buildings are in the image?"),
    "count_flooded":      ("Complex_Counting", "How many buildings are flooded?"),
    "count_nonflooded":   ("Complex_Counting", "How many buildings are non flooded?"),
    "overall_condition":  ("Condition_Recognition", "What is the overall condition of the given image?"),
    "road_condition":     ("Condition_Recognition", "What is the condition of road?"),
    "road_yn_flooded":    ("Yes_No", "Is the entire road flooded?"),
    "road_yn_nonflooded": ("Yes_No", "Is the entire road non flooded?"),
    # new intents for the other datasets
    "image_flooded_yn":   ("Yes_No", "Is the image flooded?"),
    "building_any_yn":    ("Yes_No", "Are there any buildings in the image?"),
    "building_flooded_yn": ("Yes_No", "Are any buildings flooded?"),
    "building_damage_yn": ("Yes_No", "Are any buildings damaged?"),
    "road_any_yn":        ("Yes_No", "Is there a road in the image?"),
    "road_damage_yn":     ("Yes_No", "Is any road damaged?"),
    "water_any_yn":       ("Yes_No", "Is there any water in the image?"),
    "sky_yn":             ("Yes_No", "Is there sky present?"),
    "environment":        ("Scene_Recognition", "What environment is flooded?"),
    "flood_extent":       ("Flood_Extent", "How much of the image is covered by flood water?"),
}
# flood_extent answers: share of image pixels labelled flood water
EXTENT_BINS = [(0.0, "none"), (0.10, "less than 10%"), (0.25, "10 to 25%"), (0.50, "25 to 50%"), (1.01, "more than 50%")]

def extent_answer(frac):
    if frac <= 0: return "none"
    return next(name for hi, name in EXTENT_BINS[1:] if frac < hi)

def norm(a):
    """Same normalisation as build_table.norm: lower-case, '-' -> ' ', digits -> int."""
    if a is None: return None
    s = str(a).strip().lower().replace("-", " ")
    return int(s) if s.isdigit() else s

def row(dataset, split, image_id, image, intent, answer, source, question=None, **meta):
    qt, q = INTENTS[intent]
    r = dict(dataset=dataset, split=split, image_id=image_id, image=os.path.relpath(image, ROOT),
             question=question or q, question_type=qt, intent=intent, answer=norm(answer), answer_source=source)
    if meta: r["meta"] = meta
    return r

def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

def summary(rows):
    import collections
    print(f"{len(rows)} QA pairs over {len({(r['split'], r['image_id']) for r in rows})} images")
    for (sp, it), n in sorted(collections.Counter((r["split"], r["intent"]) for r in rows).items()):
        ans = collections.Counter(r["answer"] for r in rows if r["split"] == sp and r["intent"] == it)
        print(f"  {sp:5s} {it:20s} {n:5d}  {dict(ans.most_common(4))}")
