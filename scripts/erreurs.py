"""Detail objet par objet d'un modele a un seuil donne, sur les photos annotees :
un CSV (TP / FP / FN, score max du modele) et une image par erreur
(vert = annotation, rouge = masque du modele).

python scripts/erreurs.py cache/gdino_sam 0.4 results/erreurs_gdino_0.4
"""
import argparse
import csv
import glob
import os
from collections import Counter

import numpy as np
from PIL import Image

from commun import FUSION, colonnes, lire_annotation, lire_labels, masques, regrouper

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("seuil", type=float)
ap.add_argument("out")
a = ap.parse_args()

labels = lire_labels("config/labels_isir60.txt")
os.makedirs(a.out, exist_ok=True)
lignes, fp, fn = [], Counter(), Counter()

for f in sorted(glob.glob(os.path.join("data/annotations", "*.json"))):
    nom = os.path.splitext(os.path.basename(f))[0]
    h, w, gt = lire_annotation(f)
    probs = np.load(os.path.join(a.cache, nom + ".npz"))["probs"].astype(np.float32)
    preds, gts = regrouper(labels, masques(probs, h, w, False, a.seuil, "nearest"), gt, h, w)
    score = {}
    for label, p in zip(labels, probs):
        l = FUSION.get(label, label)
        score[l] = max(score.get(l, 0.0), float(p.max()))
    photo = np.array(Image.open(f"data/test/dataset_ISIR_60/{nom}.jpg").convert("RGB"), dtype=np.float32)

    for c, p in preds.items():
        g_on, p_on = colonnes(gts[c]).any(), colonnes(p).any()
        if not (g_on or p_on):
            continue
        type_ = "TP" if g_on and p_on else ("FP" if p_on else "FN")
        lignes.append({"image": nom, "categorie": c, "type": type_, "score_max": round(score[c], 3)})
        if type_ == "TP":
            continue
        (fp if type_ == "FP" else fn)[c] += 1
        img = photo.copy()
        for m, couleur in ((gts[c], (0, 255, 0)), (p, (255, 0, 0))):
            img[m] = 0.5 * img[m] + 0.5 * np.array(couleur)
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(a.out, f"{nom}_{type_}_{c.replace(' ', '_')}.jpg"))

with open(os.path.join(a.out, "objets.csv"), "w", newline="", encoding="utf-8") as fo:
    wr = csv.DictWriter(fo, fieldnames=["image", "categorie", "type", "score_max"])
    wr.writeheader()
    wr.writerows(lignes)

print(f"{len(lignes)} objets, FP {sum(fp.values())}, FN {sum(fn.values())}")
for c in sorted(set(fp) | set(fn), key=lambda c: -(fp[c] + fn[c])):
    print(f"  {c:16s} FP {fp[c]:3d}  FN {fn[c]:3d}")
