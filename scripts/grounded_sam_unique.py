"""Grounding DINO + SAM, requete unique : toutes les categories dans une seule phrase
("bed. cabinet. chair. ..."). Chaque boite est attribuee a la categorie qui obtient le
meilleur score (moyenne sur les mots de la categorie) : les categories sont en concurrence.
Meme cache que clipseg_infer.py : sweep_clipseg.py et erreurs.py s'appliquent tels quels.

python scripts/grounded_sam_unique.py data/test/dataset_ISIR_60 cache/gdino_sam_unique
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

from commun import lire_labels

DINO = "IDEA-Research/grounding-dino-tiny"
SAM = "facebook/sam-vit-base"
SCORE_MIN = 0.1  # boites gardees pour le balayage des seuils
MAX_BOITES = 3   # par categorie

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("out")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_proc = AutoProcessor.from_pretrained(DINO)
dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(device)
sam_proc = SamProcessor.from_pretrained(SAM)
sam = SamModel.from_pretrained(SAM).to(device)
labels = lire_labels("config/labels_isir60.txt")
os.makedirs(a.out, exist_ok=True)

# Phrase unique et position des mots de chaque categorie dans la phrase tokenisee
texte = ". ".join(labels) + "."
ids = dino_proc.tokenizer(texte)["input_ids"]
tokens, k = [], 0
for l in labels:
    sous = dino_proc.tokenizer(l, add_special_tokens=False)["input_ids"]
    k = next(j for j in range(k, len(ids)) if ids[j:j + len(sous)] == sous)
    tokens.append(list(range(k, k + len(sous))))
    k += len(sous)

temps = []
for f in sorted(glob.glob(os.path.join(a.images, "*.jpg"))):
    image = Image.open(f).convert("RGB")
    W, H = image.size
    t0 = time.time()

    entrees = dino_proc(images=image, text=texte, return_tensors="pt").to(device)
    with torch.no_grad():
        sorties = dino(**entrees)
    proba = sorties.logits[0].sigmoid().cpu()                            # (900 boites, 256 tokens)
    score = torch.stack([proba[:, t].mean(1) for t in tokens], 1)        # (900, n categories)
    s, cat = score.max(1)
    cx, cy, bw, bh = sorties.pred_boxes[0].cpu().unbind(1)
    xyxy = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], 1)

    boites, scores, cats = [], [], []
    for i in range(len(labels)):
        idx = ((cat == i) & (s >= SCORE_MIN)).nonzero()[:, 0]
        for j in idx[s[idx].argsort(descending=True)[:MAX_BOITES]]:
            boites.append(xyxy[j].tolist())
            scores.append(s[j].item())
            cats.append(i)

    probs = np.zeros((len(labels), H, W), np.float32)
    if boites:
        entrees = sam_proc(image, input_boxes=[boites], return_tensors="pt").to(device)
        with torch.no_grad():
            sorties = sam(**entrees, multimask_output=False)
        masques = sam_proc.image_processor.post_process_masks(
            sorties.pred_masks.cpu(), entrees["original_sizes"].cpu(), entrees["reshaped_input_sizes"].cpu())[0]
        for m, sc, i in zip(masques[:, 0].numpy(), scores, cats):
            probs[i] = np.maximum(probs[i], m * sc)

    temps.append(time.time() - t0)
    nom = os.path.splitext(os.path.basename(f))[0]
    np.savez_compressed(os.path.join(a.out, nom + ".npz"), probs=probs.astype(np.float16), h=H, w=W)
    print(nom, f"{temps[-1]:.2f} s, {len(boites)} boites")

print(f"temps moyen : {np.mean(temps):.2f} s/image ({device})")
