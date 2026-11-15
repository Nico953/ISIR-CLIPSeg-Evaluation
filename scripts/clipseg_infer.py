"""CLIPSeg sur chaque photo avec les 31 categories comme prompts.
Enregistre les cartes de probabilite brutes (352x352) : seuil et normalisation sont
appliques ensuite par sweep_clipseg.py, sans relancer le modele.

python scripts/clipseg_infer.py data/test/dataset_ISIR_60 cache/nom
python scripts/clipseg_infer.py data/test/dataset_ISIR_60 cache/photo --template "a photo of a {}."
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from commun import lire_labels

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("out")
ap.add_argument("--template", default="{}")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
labels = lire_labels("config/labels_isir60.txt")
prompts = [a.template.format(l) for l in labels]
os.makedirs(a.out, exist_ok=True)

temps = []
for f in sorted(glob.glob(os.path.join(a.images, "*.jpg"))):
    image = Image.open(f).convert("RGB")
    t0 = time.time()
    entrees = processor(text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(**entrees, interpolate_pos_encoding=True).logits).cpu().numpy()
    temps.append(time.time() - t0)
    nom = os.path.splitext(os.path.basename(f))[0]
    np.savez_compressed(os.path.join(a.out, nom + ".npz"), probs=probs.astype(np.float16), h=image.height, w=image.width)
    print(nom, f"{temps[-1]:.2f} s")

print(f"temps moyen : {np.mean(temps):.2f} s/image ({device})")
