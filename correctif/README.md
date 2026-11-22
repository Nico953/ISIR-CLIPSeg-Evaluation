# Correctif de garbage.py (lekiwi-ros2)

- `garbage_original.py` : version d'origine de lekiwi-ros2
- `garbage.py` : version corrigee
- `garbage_correctif.diff` : les changements

Changements :
1. Plus de normalisation min-max avant le seuil (elle produisait un masque meme pour un objet absent).
2. Seuil 0,3 sur la probabilite CLIPSeg (`SEUIL`). Plateau de F1 entre 0,2 et 0,35 en mode pipeline.
3. CLIPSeg recoit le nom seul ("fridge"), le .npz garde le label complet ("fridge: large metal fridge") : projector.py inchange.
4. Cas a un seul prompt (CLIPSeg renvoie alors une carte 2D).

Resultats (59 photos ISIR_60, mode pipeline, prompts de Gemma) : F1 57,5 -> 64 %, IoU colonnes 21 -> 34 %.

Pour l'utiliser sur le robot : remplacer garbage.py dans le workspace lekiwi-ros2 par `garbage.py`, puis relancer garbage.py et projector.py.
Non encore teste sur le robot.
