import os
import re
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from time import time
import json
import ollama
import ast

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
output_folder = 'masks'

# LABEL_CREATION_PROMPT = (
#     "You're mounted on a mobile robot and looking basically from the ground level a bit up. "
#     "Be as specific as possible, name all the objects you can see on this image. "
#     "If you want to say ceiling/floor/wall, some sort of a construction which"
#     "is not identifiable, ignore it, there's no point of not precise descriptions"
#     "Output your response as a python list."
# )
LABEL_CREATION_PROMPT = "You're mounted on a mobile robot and looking from the ground level a bit up. Be as specific as possible, name the objects you can see and recognize on this image. Ignore walls/ceilings/floors. Ignore the objects located on the ceiling/high up, ex: lamps, ventilation pipes, etc. Ignore very small objects, like lightswitches, wall markings, door handles. Your descriptions should be very precise and name the object (try to avoid generic descriptions as “industrial looking environment/modern architecture, etc”). If there’s multiple objects under the same descriptions (multiple chairs/windows…) provide a generic description once for one of them. Format your descriptions as “item name: a short description”, ex: \"fridge: large metal fridge\" Output your response as a python list."

files = []

for file in os.listdir("./captured_images//"):
    if '.jpg'in file:
        files.append(
            (f"captured_images/{file}", f"./captured_images/{file.replace('jpg', 'json')}")
        )


for img_file, json_file in files:

    image = Image.open(img_file)

    # with open(json_file, "r") as f:
    #     mask_meta = json.load(f)
    #     prompts = mask_meta["labels_used"]
    response = ollama.chat(
        model="gemma4:12b",
        messages=[
            {
                "role": "user",
                "content": LABEL_CREATION_PROMPT,
                "images": [img_file],
            }
        ],
    )
    
    raw = response["message"]["content"]
    clean = re.sub(r"```(?:python)?\s*|\s*```", "", raw).strip()
    
    try:
        parsed = ast.literal_eval(clean)
    except (ValueError, SyntaxError):
        parsed = re.findall(r"['\"]([^'\"]+)['\"]", clean)
    
    if not parsed:
        print(f"Could not parse labels for {img_file}, skipping.")
        continue
    
    labels = parsed

    now = time()
    prompts = [p.replace("/", " or ") for p in labels]
    inputs = processor(
        text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits  # shape: (num_prompts, H, W)
    preds = torch.sigmoid(logits)
    
    orig_w, orig_h = image.size
    
    masks_to_save = {}
    for i, prompt in enumerate(prompts):
        mask = preds[i].numpy()
        # Normalize to 0-1
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        # Threshold to binary mask
        binary_mask = (mask > 0.9).astype(np.uint8)  # 1 where predicted, 0 elsewhere
        # Resize mask to original image size
        mask_img = Image.fromarray(binary_mask * 255).resize(
            (orig_w, orig_h),
            resample=Image.NEAREST,  # NEAREST preserves hard edges
        )
        mask_resized = np.array(mask_img).astype(np.float32) / 255.0
        # Apply a single color to the mask (e.g. red)
        colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        colored_mask[mask_resized > 0.5] = [255, 0, 0]  # red for masked region
        # Blend with original image
        # overlay = Image.blend(image, Image.fromarray(colored_mask), alpha=0.5)
    
        # overlay.save(f"overlay_{prompt.replace(' ', '_')}.png")
        boolean_mask = mask_resized > 0.5  # shape: (orig_h, orig_w), dtype: bool

        masks_to_save[f"mask_{i}_id"] = np.array([i])
        masks_to_save[f"mask_{i}_prompt"] = np.array([prompt])
        masks_to_save[f"mask_{i}_map"] = boolean_mask

    img_basename = os.path.splitext(os.path.basename(img_file))[0]

    output_path = os.path.join(output_folder, f"{img_basename}.npz")
    np.savez(output_path, **masks_to_save)


    
    print(f"time of the inference: {time() - now}")