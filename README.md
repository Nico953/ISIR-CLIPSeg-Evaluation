# ISIR-CLIPSeg-Evaluation

Mission 2 : reduire les faux positifs et faux negatifs de CLIPSeg dans le pipeline LeKiwi (VLM -> CLIPSeg -> projector.py). Photos ISIR_60 de la mission 1.

## Annotation
```
pip install labelme
labelme data/test/dataset_ISIR_60 --output data/annotations --labels config/labels_isir60.txt --validatelabel exact --nodata --autosave
```
Outil AI-Polygon (SAM). Annoter toutes les categories de la liste visibles sur la photo, meme label pour toutes les instances.

## Commandes
```
python scripts/clipseg_infer.py data/test/dataset_ISIR_60 cache/nom
python scripts/sweep_clipseg.py cache/nom results/sweep_nom.csv
```
`norm_s0.9_nearest` = reglage actuel de garbage.py.
