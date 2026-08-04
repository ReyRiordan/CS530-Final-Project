import os
import random
from pathlib import Path
import numpy as np
from PIL import Image

from src.paths import DATA_DIR


# ----- CONFIG -----

N_TO_GENERATE = 3000  # num data points to generate
VAL_SPLIT = 0.1

# Exponent base B for cross side placeement probability weight: P = B^d, where d = rows from center
# Ex. First row across -> P = 0.8^0 = 1, last row across -> P = 0.8^15 = ~0.03
CROSS_SIDE_BASE = 0.9

# Scaling noise: scale = troop scale * random(1 - noise, 1 + noise)
NOISE_SCALE = 0.1
# Std of pixels added to RGB
NOISE_PIXEL = 5.0

# Scale factor to apply to troop sprites (to try to match actual in-game size)
TROOP_SCALES = {
    "archer": 0.7,
    "giant": 1.4,
    "goblin": 0.9,
    "knight": 1.0,
    "mekka": 0.9,
    "minion": 0.7,
    "musketeer": 1.0,
    "spoblin": 1.0,
}

# Class defs (ORDER MATTERS)
TROOP_NAMES = ["archer", "giant", "goblin", "knight", "mekka", "minion", "musketeer", "spoblin"]
SIDES = ["ally", "enemy"]

# Pixel bounds of actual tile grid within the base arena image
ARENA_TILE_BOUNDS = (0, 0, 840, 1215)

# Arena mask for valid troop placement (true = can place on that tile)
PLACEMENT_MASK = np.array([
    [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
], dtype=bool)

# Paths
SPRITES_DIR = str(DATA_DIR / "sprites")
ARENA_TEMPLATE = str(Path(__file__).resolve().parent / "templates" / "arena_default.jpg")
OUTPUT_DIR = str(DATA_DIR / "synthetic_dataset")

# Useful later
NUM_CLASSES = 16
CLASS_NAMES = [f"{troop}_{side}" for side in SIDES for troop in TROOP_NAMES] # Class ID -> sprite folder
CLASS_INFO = [(troop, side) for side in SIDES for troop in TROOP_NAMES] # Class ID -> (troop_name, side)


# ----------


def compute_placement_weights(valid_coords, is_ally):
    """Calc weights per tile, see CROSS_SIDE_BASE comments"""
    weights = np.ones(len(valid_coords), dtype=float)
    for i, (r, _) in enumerate(valid_coords):
        if is_ally and r < 16:
            weights[i] = CROSS_SIDE_BASE ** (15-r)
        elif not is_ally and r >= 16:
            weights[i] = CROSS_SIDE_BASE ** (r-16)
        # else leave at prob 1

    return weights


def build_tile_centers(bounds = ARENA_TILE_BOUNDS):
    """Return (32, 18, 2) array of (x, y) pixel centers for each tile."""
    x_min, y_min, x_max, y_max = bounds
    tile_w = (x_max-x_min) / 18
    tile_h = (y_max-y_min) / 32
    rows, cols = np.arange(32), np.arange(18)
    x_centers = x_min + ((cols+0.5) * tile_w)
    y_centers = y_min + ((rows+0.5) * tile_h)
    centers = np.stack(np.meshgrid(x_centers, y_centers), axis=-1) # stack along coord axis so we can do centers[r,c] = (x,y)

    return centers


def index_sprites(sprites_dir):
    """Creates index {class_name: list[sprite_paths]}"""
    index = {}
    for class_name in CLASS_NAMES:
        folder = os.path.join(sprites_dir, class_name)
        sprites = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png"))
        index[class_name] = sprites
    
    return index


def add_scale_noise(sprite, troop_name):
    """Add scaling noise to sprite"""
    base_scale = TROOP_SCALES[troop_name]
    scale = base_scale * random.uniform(1.0-NOISE_SCALE, 1.0+NOISE_SCALE)
    sprite_h, sprite_w = sprite.shape[:2]
    new_w = int(round(sprite_w * scale))
    new_h = int(round(sprite_h * scale))
    sprite_pil = Image.fromarray(sprite).resize((new_w, new_h), Image.LANCZOS)

    return np.array(sprite_pil, dtype=np.uint8)


def add_pixel_noise(sprite):
    """Add pixel noise to sprite RGB, exclude A (alpha)"""
    noise = np.random.normal(0, NOISE_PIXEL, sprite[:, :, :3].shape)
    result = sprite.copy()
    result[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result


def paste_sprite(arena, sprite, center_x, center_y):
    """Paste sprite with center point on (center_x, center_y), return tight bounding box"""
    sprite_h, sprite_w = sprite.shape[:2]
    arena_h, arena_w = arena.shape[:2]

    # Sprite top left corner
    paste_x = center_x - sprite_w // 2
    paste_y = center_y - sprite_h // 2

    # Overlap coords (to cut off any sprite part hanging off arena)
    arena_x1 = max(paste_x, 0)
    arena_y1 = max(paste_y, 0)
    arena_x2 = min(paste_x + sprite_w, arena_w)
    arena_y2 = min(paste_y + sprite_h, arena_h)
    sprite_x1 = arena_x1 - paste_x
    sprite_y1 = arena_y1 - paste_y
    sprite_x2 = sprite_x1 + (arena_x2 - arena_x1)
    sprite_y2 = sprite_y1 + (arena_y2 - arena_y1)

    if arena_x1 >= arena_x2 or arena_y1 >= arena_y2: return None # somehow completely outside arena

    # Crop parts relevant for pasting
    sprite_crop = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
    arena_crop = arena[arena_y1:arena_y2, arena_x1:arena_x2]

    # Paste in crop, then replace actual cropped region in arena
    sprite_alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0
    pasted = (sprite_crop[:, :, :3].astype(np.float32) * sprite_alpha) + (arena_crop[:, :, :3].astype(np.float32) * (1-sprite_alpha))
    arena[arena_y1:arena_y2, arena_x1:arena_x2, :3] = np.clip(pasted, 0, 255).astype(np.uint8)

    # Update arena alpha
    arena_alpha = arena_crop[:, :, 3].astype(np.float32) / 255.0
    new_alpha = sprite_alpha[:, :, 0] + arena_alpha * (1 - sprite_alpha[:, :, 0])
    arena[arena_y1:arena_y2, arena_x1:arena_x2, 3] = np.clip(new_alpha * 255, 0, 255).astype(np.uint8)

    # Create tight bounding box based on first visible sprite pixels
    visible = sprite_crop[:, :, 3] >= 10 # alpha threshold
    if not visible.any(): return None
    rows_vis = np.where(visible.any(axis=1))[0]
    cols_vis = np.where(visible.any(axis=0))[0]
    bbox_x1 = arena_x1 + int(cols_vis[0])
    bbox_y1 = arena_y1 + int(rows_vis[0])
    bbox_x2 = arena_x1 + int(cols_vis[-1]) + 1
    bbox_y2 = arena_y1 + int(rows_vis[-1]) + 1

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2


# ----------


def generate():
    # Setup output dirs
    train_image_dir = os.path.join(OUTPUT_DIR, "train", "images")
    train_label_dir = os.path.join(OUTPUT_DIR, "train", "labels")
    val_image_dir = os.path.join(OUTPUT_DIR, "val", "images")
    val_label_dir = os.path.join(OUTPUT_DIR, "val", "labels")
    for d in (train_image_dir, train_label_dir, val_image_dir, val_label_dir):
        os.makedirs(d, exist_ok=True)

    # Setup
    base_arena = np.array(Image.open(ARENA_TEMPLATE).convert("RGBA"), dtype=np.uint8)
    image_h, image_w = base_arena.shape[:2]
    tile_centers = build_tile_centers(ARENA_TILE_BOUNDS)
    sprite_index = index_sprites(SPRITES_DIR)
    valid_coords = np.argwhere(PLACEMENT_MASK)  # turn mask into array of valid (r,c) coords

    # Start if/where we left off
    n_train = len([f for f in os.listdir(train_image_dir) if f.endswith(".jpg")])
    n_val = len([f for f in os.listdir(val_image_dir) if f.endswith(".jpg")])
    n_existing = n_train + n_val
    for i in range(n_existing, n_existing + N_TO_GENERATE):
        arena = base_arena.copy()
        labels = [] # list[class_id, bounding box]

        # Select total num of troops to paste
        num_troops = random.randint(1, 10)

        for _ in range(num_troops):
            # Select which troops to paste
            class_id = random.randint(0, NUM_CLASSES - 1)
            troop_name, side = CLASS_INFO[class_id]
            is_ally = (side == "ally")
            
            # Select tile to paste
            weights = compute_placement_weights(valid_coords, is_ally)
            idx = random.choices(range(len(valid_coords)), weights=weights, k=1)[0]
            tile_row, tile_col = valid_coords[idx]
            center_x, center_y = tile_centers[tile_row, tile_col]
            center_x, center_y = int(round(center_x)), int(round(center_y))

            # Select sprite to paste
            frame_path = random.choice(sprite_index[f"{troop_name}_{side}"])
            sprite = np.array(Image.open(frame_path).convert("RGBA"), dtype=np.uint8)

            # Add noise
            sprite = add_scale_noise(sprite, troop_name)
            sprite = add_pixel_noise(sprite)

            # Paste + get tight bounding box
            bbox = paste_sprite(arena, sprite, center_x, center_y)
            if bbox is not None:
                labels.append((class_id, *bbox))

        # Assign to val or train split
        is_val = random.random() < VAL_SPLIT
        if is_val:
            split_image_dir, split_label_dir = val_image_dir, val_label_dir
            split_idx = n_val
            n_val += 1
        else:
            split_image_dir, split_label_dir = train_image_dir, train_label_dir
            split_idx = n_train
            n_train += 1

        # Save image
        img_path = os.path.join(split_image_dir, f"{split_idx:06d}.jpg")
        Image.fromarray(arena[:, :, :3]).save(img_path, quality=95) # jpeg standards

        # Write YOLO labels: class_id, centers, bbox dims (all normalized)
        label_path = os.path.join(split_label_dir, f"{split_idx:06d}.txt")
        with open(label_path, "w") as f:
            for class_id, x1, y1, x2, y2 in labels:
                center_x_norm = ((x1+x2) / 2) / image_w
                center_y_norm = ((y1+y2) / 2) / image_h
                bbox_w_norm = (x2-x1) / image_w
                bbox_h_norm = (y2-y1) / image_h
                f.write(f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f}\n")

        if (i+1) % 100 == 0:
            print(f"[{i+1}/{n_existing + N_TO_GENERATE}] saved to {img_path}")


if __name__ == "__main__":
    generate()
