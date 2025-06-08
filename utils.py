import numpy as np
from PIL import Image
import io
import os
import cv2


# Mapping Cityscapes → 8 classes
label_mapping = {
    0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7,
    7: 0, 8: 0, 9: 0, 10: 0,
    11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3,
    17: 4, 18: 4, 19: 4, 20: 4,
    21: 5, 22: 5,
    23: 6,
    24: 1, 25: 1,
    26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2
}


# Liste de couleurs
colors = [
    (100, 100, 100),   # 0: route
    (200, 0, 0),       # 1: humain
    (0, 0, 200),       # 2: véhicule
    (255, 255, 0),     # 3: construction
    (50, 50, 0),       # 4: objet
    (0, 200, 0),       # 5: nature
    (150, 200, 255),   # 6: ciel
    (255, 255, 255)    # 7: void
]


def map_labels(mask):
    mapped = np.full_like(mask, fill_value=7)
    for original, new in label_mapping.items():
        mapped[mask == original] = new
    return mapped


def mask_to_color(mask):
    """
    Convertit un masque 2D (0 à 7) en image RGB colorisée.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(colors):
        color_mask[mask == class_id] = color

    return Image.fromarray(color_mask)


def preprocess_image_bytes(image_bytes, target_size=(256, 512)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size[::-1])
    img_np = np.array(image) / 255.0
    return np.expand_dims(img_np, axis=0)


def load_real_image(image_id, base_dir="../data/images/val"):
    """
    Charge une image réelle depuis son ID
    """
    relative_path = f"{image_id}.png"
    full_path = os.path.normpath(os.path.join(base_dir, relative_path))

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image non trouvée : {full_path}")

    img = Image.open(full_path).convert("RGB")
    img_resized = img.resize((512, 256))
    return img_resized


def load_real_mask(image_id, base_dir="../data/masks/val"):
    image_id = image_id.replace("leftImg8bit", "gtFine_labelIds")
    relative_path = f"{image_id}.png"
    full_path = os.path.normpath(os.path.join(base_dir, relative_path))

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Masque non trouvé : {full_path}")

    mask = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Erreur lors de la lecture du masque : {full_path}")

    mask_resized = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_NEAREST)
    mapped_mask = map_labels(mask_resized)

    return Image.fromarray(mapped_mask.astype(np.uint8))
