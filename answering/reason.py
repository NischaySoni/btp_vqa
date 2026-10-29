"""Scene description + intent -> answer, with one fitted parameter per intent.
Each intent reads one scalar from the scene (FEATURES, sometimes a choice of two); fit() picks the feature and threshold (or, for counts, the min instance
area and optional merged-blob splitting) that maximises accuracy on the training rows only. answer() is deterministic, so every prediction can be traced
back to "share of flooded road = 0.31 > 0.18" or "4 flooded building components >= 0.02% of the image"."""
import numpy as np

def _f(s, *names): return sum(s["frac"].get(n, 0.0) for n in names)
def _share(a, b): return a / (a + b) if a + b > 0 else -1.0          # -1: class absent

FLOODED = lambda s: _f(s, "building_flooded", "road_flooded", "flood")
FLOODED_W = lambda s: _f(s, "building_flooded", "road_flooded", "flood", "water")   # generic water counted as flood
FEATURES = {  # intent -> candidate scene -> scalar maps; fit() picks (candidate, threshold); yes when scalar > threshold
    "overall_condition":   [FLOODED, FLOODED_W],
    "image_flooded_yn":    [FLOODED, FLOODED_W],
    "road_condition":      [lambda s: _share(_f(s, "road_flooded"), _f(s, "road_nonflooded"))],
    "road_yn_flooded":     [lambda s: _share(_f(s, "road_flooded"), _f(s, "road_nonflooded"))],
    "road_yn_nonflooded":  [lambda s: _share(_f(s, "road_nonflooded"), _f(s, "road_flooded"))],
    "sky_yn":              [lambda s: _f(s, "sky")],
    "environment":         [lambda s: _f(s, "building", "building_flooded", "building_nonflooded")],
    "building_any_yn":     [lambda s: _f(s, "building", "building_flooded", "building_nonflooded")],
    "road_any_yn":         [lambda s: _f(s, "road_flooded", "road_nonflooded")],
    "water_any_yn":        [lambda s: _f(s, "water", "flood", "road_flooded", "building_flooded", "pool")],
}
LABELS = {"overall_condition": ("flooded", "non flooded"), "road_condition": ("flooded", "non flooded"),
          "environment": ("urban", "rural")}                       # (above threshold, below); default yes / no
COUNTS = {"count_total": lambda b: b, "count_flooded": lambda b: [x for x in b if x[1]],
          "count_nonflooded": lambda b: [x for x in b if not x[1]]}
AREAS = [0, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]   # min instance share of the image
# merged-blob splitting: a component larger than t x (q-th percentile component area in this image) counts as
# round(area / that reference) buildings; (None, None) = plain connected components
SPLITS = [(None, None)] + [(q, t) for q in (75, 90) for t in (1.5, 2, 3)]
COUNT_PARAMS = [(a, q, t) for a in AREAS for q, t in SPLITS]
DERIVED = {"building_flooded_yn": "count_flooded"}              # yes iff that count > 0

def count(scene, intent, p):
    min_area, q, t = p if isinstance(p, (tuple, list)) else (p, None, None)
    buildings = scene.get("buildings_flooded_ws", scene["buildings"]) if intent == "count_flooded" else scene["buildings"]
    sel = [a for a, _ in COUNTS[intent](buildings) if a >= min_area]
    if q is None or not sel: return len(sel)
    ref = np.percentile([a for a, _ in buildings if a >= min_area], q)
    return int(sum(max(1, round(a / ref)) if a > t * ref else 1 for a in sel))

def supported(intent): return intent in COUNTS or intent in DERIVED or intent in FEATURES

def answer(scene, intent, p):
    if intent in COUNTS: return count(scene, intent, p)
    if intent in DERIVED: return "yes" if count(scene, DERIVED[intent], p) > 0 else "no"
    hi, lo = LABELS.get(intent, ("yes", "no")); j, thr = p
    return hi if FEATURES[intent][j](scene) > thr else lo

def fit(scenes, answers, intent):
    """Best parameter on the given training rows (ties -> first, i.e. the smaller threshold / area)."""
    if intent in COUNTS or intent in DERIVED:
        cands = COUNT_PARAMS
    else:
        cands = []
        for j, feat in enumerate(FEATURES[intent]):
            v = np.unique([feat(s) for s in scenes]); cands += [(j, t) for t in list((v[:-1] + v[1:]) / 2) + [-0.5, 1.0]]
    acc = [np.mean([answer(s, intent, c) == a for s, a in zip(scenes, answers)]) for c in cands]
    return cands[int(np.argmax(acc))]
