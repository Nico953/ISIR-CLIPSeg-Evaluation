import json

import numpy as np
from PIL import Image, ImageDraw


def lire_labels(chemin):
    return [l.strip() for l in open(chemin, encoding="utf-8") if l.strip()]


def lire_annotation(chemin):
    """Annotation LabelMe -> (h, w, {label: masque}). Les instances d'un label sont reunies."""
    a = json.load(open(chemin, encoding="utf-8"))
    h, w = a["imageHeight"], a["imageWidth"]
    calques = {}
    for s in a["shapes"]:
        img = calques.setdefault(s["label"], Image.new("L", (w, h)))
        pts = [tuple(p) for p in s["points"]]
        if s["shape_type"] == "rectangle":
            (x0, y0), (x1, y1) = pts
            ImageDraw.Draw(img).rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
        else:
            ImageDraw.Draw(img).polygon(pts, fill=255)
    return h, w, {k: np.array(v) > 0 for k, v in calques.items()}


def colonnes(masque):
    """Colonnes ou projector.py lance un rayon : moitie basse, au moins 10 pixels."""
    return masque[masque.shape[0] // 2:].sum(axis=0) >= 10


def masques(probs, h, w, norm, seuil, methode):
    """probs (N, 352, 352) -> masques (N, h, w).
    norm=True, seuil=0.9, methode='nearest' reproduit garbage.py."""
    rs = Image.NEAREST if methode == "nearest" else Image.BILINEAR
    grandes = np.stack([np.array(Image.fromarray(p, mode="F").resize((w, h), rs)) for p in probs])
    if norm:
        mn = probs.min(axis=(1, 2))[:, None, None]
        mx = probs.max(axis=(1, 2))[:, None, None]
        grandes = (grandes - mn) / (mx - mn + 1e-8)
    return grandes > seuil


def iou(a, b):
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()
