"""Compare les reglages de post-traitement de CLIPSeg sur les photos annotees :
normalisation min-max (oui/non) x seuil x agrandissement (nearest/bilineaire).
La config norm_s0.9_nearest est celle de garbage.py.

Un objet est detecte si projector.py lance au moins un rayon pour lui.
Une categorie absente de l'annotation est consideree absente de la photo.

python scripts/sweep_clipseg.py cache/nom results/sweep_nom.csv
"""
import argparse
import csv
import glob
import os

import numpy as np

from commun import colonnes, iou, lire_annotation, lire_labels, masques

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("sortie")
a = ap.parse_args()

labels = lire_labels("config/labels_isir60.txt")
configs = [(n, s, m) for n in (True, False) for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
           for m in ("nearest", "bilineaire")]
stats = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "iou_2d": [], "iou_col": []} for c in configs}

n_images = 0
for f in sorted(glob.glob(os.path.join("data/annotations", "*.json"))):
    nom = os.path.splitext(os.path.basename(f))[0]
    h, w, gt = lire_annotation(f)
    z = np.load(os.path.join(a.cache, nom + ".npz"))
    probs = z["probs"].astype(np.float32)
    n_images += 1
    for c in configs:
        pred = masques(probs, h, w, *c)
        st = stats[c]
        for label, p in zip(labels, pred):
            g = gt.get(label, np.zeros((h, w), bool))
            g_col, p_col = colonnes(g), colonnes(p)
            if g_col.any():
                st["TP" if p_col.any() else "FN"] += 1
                st["iou_2d"].append(iou(g, p))
                st["iou_col"].append(iou(g_col, p_col))
            else:
                st["FP" if p_col.any() else "TN"] += 1

lignes = []
for (n, s, m), st in stats.items():
    tp, fp, fn, tn = st["TP"], st["FP"], st["FN"], st["TN"]
    lignes.append({
        "config": f"{'norm' if n else 'brut'}_s{s}_{m}",
        "precision": tp / (tp + fp) if tp + fp else 0,
        "rappel": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
        "taux_fp": fp / (fp + tn) if fp + tn else 0,
        "miou_2d": np.mean(st["iou_2d"]),
        "miou_colonnes": np.mean(st["iou_col"]),
        "TP": tp, "FP": fp, "FN": fn,
    })
lignes.sort(key=lambda l: -l["f1"])

os.makedirs(os.path.dirname(a.sortie) or ".", exist_ok=True)
with open(a.sortie, "w", newline="", encoding="utf-8") as fo:
    wr = csv.DictWriter(fo, fieldnames=list(lignes[0]))
    wr.writeheader()
    wr.writerows(lignes)

print(f"{n_images} images annotees")
print(f"{'config':24s} {'P':>6s} {'R':>6s} {'F1':>6s} {'tauxFP':>7s} {'mIoU2D':>7s} {'mIoUcol':>7s}")
for l in lignes:
    print(f"{l['config']:24s} {l['precision']:6.1%} {l['rappel']:6.1%} {l['f1']:6.1%} {l['taux_fp']:7.1%} "
          f"{l['miou_2d']:7.1%} {l['miou_colonnes']:7.1%}")
