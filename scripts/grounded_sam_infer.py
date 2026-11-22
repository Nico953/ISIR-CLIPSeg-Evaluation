"""Grounding DINO + SAM : Grounding DINO cherche une boite par categorie, SAM trace le masque.
Pour chaque categorie, carte = max(score de la boite x masque SAM) : seuiller la carte a s
revient a garder les boites de score > s. Meme cache que clipseg_infer.py : sweep_clipseg.py
s'applique tel quel (lire les lignes brut_ ; norm_ n'a pas de sens ici).

python scripts/grounded_sam_infer.py data/test/dataset_ISIR_60 cache/gdino_sam
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

temps = []
for f in sorted(glob.glob(os.path.join(a.images, "*.jpg"))):
    image = Image.open(f).convert("RGB")
    t0 = time.time()

    boites, scores, cats = [], [], []
    for i, label in enumerate(labels):
        entrees = dino_proc(images=image, text=label + ".", return_tensors="pt").to(device)
        with torch.no_grad():
            sorties = dino(**entrees)
        r = dino_proc.post_process_grounded_object_detection(
            sorties, entrees["input_ids"], box_threshold=SCORE_MIN, text_threshold=0.0,
            target_sizes=[image.size[::-1]])[0]
        for k in r["scores"].argsort(descending=True)[:MAX_BOITES]:
            boites.append(r["boxes"][k].tolist())
            scores.append(r["scores"][k].item())
            cats.append(i)

    probs = np.zeros((len(labels), image.height, image.width), np.float32)
    if boites:
        entrees = sam_proc(image, input_boxes=[boites], return_tensors="pt").to(device)
        with torch.no_grad():
            sorties = sam(**entrees, multimask_output=False)
        masques = sam_proc.image_processor.post_process_masks(
            sorties.pred_masks.cpu(), entrees["original_sizes"].cpu(), entrees["reshaped_input_sizes"].cpu())[0]
        for m, s, i in zip(masques[:, 0].numpy(), scores, cats):
            probs[i] = np.maximum(probs[i], m * s)

    temps.append(time.time() - t0)
    nom = os.path.splitext(os.path.basename(f))[0]
    np.savez_compressed(os.path.join(a.out, nom + ".npz"), probs=probs.astype(np.float16), h=image.height, w=image.width)
    print(nom, f"{temps[-1]:.2f} s, {len(boites)} boites")

print(f"temps moyen : {np.mean(temps):.2f} s/image ({device})")
