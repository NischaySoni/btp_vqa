"""Label map -> structured scene description (the "visual context" of Prompt-RSVQA, but explicit and inspectable).
scene = dict(frac={name: share of valid pixels}, buildings=[(area_share, flooded), ...], hw=(h, w))
Two label conventions:
- FloodNet (10 classes, seg_pipeline.CLASSES order, 255 = ignore): building instances are connected components of
  building_flooded / building_nonflooded, flooded taken from the class.
- AIFloodSense masks (grey 0 background, 255 flood, 170 sky, 85 building): building instances are connected components
  of the building class; an instance is flooded if flood water lies within `touch` (share of image width) of it.
Instance areas are image shares, so one min-area threshold works at any resolution."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

FLOODNET = ["background", "building_flooded", "building_nonflooded", "road_flooded", "road_nonflooded",
            "water", "tree", "vehicle", "pool", "grass"]
AIF = {0: "background", 255: "flood", 170: "sky", 85: "building"}

def _instances(mask, valid):
    lab, n = ndi.label(mask)
    return np.bincount(lab.ravel())[1:] / valid if n else np.zeros(0), lab

def floodnet_scene(lab):
    lab = np.asarray(lab); ok = lab != 255; valid = max(ok.sum(), 1)
    frac = {c: float((lab == i).sum() / valid) for i, c in enumerate(FLOODNET)}
    b = []
    for i, fl in ((1, True), (2, False)):
        areas, _ = _instances(lab == i, valid); b += [(float(a), fl) for a in areas]
    return dict(frac=frac, buildings=b, hw=lab.shape)

def aif_scene(mask, touch=0.01):
    m = np.asarray(mask); valid = m.size
    frac = {c: float((m == v).sum() / valid) for v, c in AIF.items()}
    areas, lab = _instances(m == 85, valid)
    near = ndi.binary_dilation(m == 255, iterations=max(1, int(touch * m.shape[1])))
    flooded = np.bincount(lab[near].ravel(), minlength=len(areas) + 1)[1:] > 0 if len(areas) else []
    return dict(frac=frac, buildings=[(float(a), bool(f)) for a, f in zip(areas, flooded)], hw=m.shape)

def load(path, kind):
    im = Image.open(path)
    return floodnet_scene(im) if kind == "floodnet" else aif_scene(im)
