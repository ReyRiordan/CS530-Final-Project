import re
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import easyocr

from capture_images import process_screenshot


# ----- CONFIG -----

TROOP_NAMES = ["archer", "giant", "goblin", "knight", "mekka", "minion", "musketeer", "spoblin"]
NUM_CLASSES = 16
GRID_ROWS = 32
GRID_COLS = 18
ARENA_TILE_BOUNDS = (0, 0, 840, 1215)

SSIM_THRESHOLD = 0.9

TEMPLATES_DIR = Path("templates")
YOLO_PATH = "YOLO26n_best.pt"

# Order matters for flat_vector
TOWER_KEYS = [
    "tower_hp_ally_left",
    "tower_hp_ally_right",
    "tower_hp_ally_king",
    "tower_hp_enemy_left",
    "tower_hp_enemy_right",
    "tower_hp_enemy_king",
]
TOWER_HP_MAX = [1512, 1512, 2568, 1512, 1512, 2568]

SCREEN_NAMES = ["main", "3bars", "startconfirm", "loading", "start1", "start2", "match", "matchover", "3crownwin", "3crownloss", "end"]

# Color + tolerance to isolate only the digits for custom OCR thresholding
ALLY_TOWER_OCR_COLOR = ((190, 215, 230), 20)
ENEMY_TOWER_OCR_COLOR = ((240, 210, 235), 20)
TOWER_OCR_COLORS = {
    "tower_hp_ally_left": ALLY_TOWER_OCR_COLOR,
    "tower_hp_ally_right": ALLY_TOWER_OCR_COLOR,
    "tower_hp_ally_king": ALLY_TOWER_OCR_COLOR,
    "tower_hp_enemy_left": ENEMY_TOWER_OCR_COLOR,
    "tower_hp_enemy_right": ENEMY_TOWER_OCR_COLOR,
    "tower_hp_enemy_king": ENEMY_TOWER_OCR_COLOR,
}

# Templates for when tower dead
DEAD_TEMPLATES = {
    "tower_hp_enemy_left": "enemy_left_dead.jpg",
    "tower_hp_enemy_right": "enemy_right_dead.jpg",
    "tower_hp_ally_left": "ally_left_dead.jpg",
    "tower_hp_ally_right": "ally_right_dead.jpg",
}

# Templates for when king tower full (HP hidden)
FULL_KING_TEMPLATES = {
    "tower_hp_enemy_king": "enemy_king_full.jpg",
    "tower_hp_ally_king": "ally_king_full.jpg",
}


# ----- CACHING -----

YOLO_LOADED = False
YOLO_MODEL = None
CARD_TEMPLATES = None
ELIXIR_TEMPLATES = None
TOWER_DEAD_TEMPLATES = None
TOWER_FULL_TEMPLATES = None
SCREEN_TEMPLATES = None
OCR_READER = None


# ----- LOAD GREYSCALE TEMPLATES -----

def to_grayscale(image):
    """Grayscale, works with PIL Image or np array"""
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    return np.array(image.convert("L"))


def load_screen_templates():
    return {name: to_grayscale(Image.open(TEMPLATES_DIR / f"screen_{name}.jpg")) for name in SCREEN_NAMES}


def load_card_templates():
    return {troop: to_grayscale(Image.open(TEMPLATES_DIR / f"card_{troop}.jpg")) for troop in TROOP_NAMES}


def load_tower_templates():
    """Dead princess towers and full king towers"""
    dead = {key: to_grayscale(Image.open(TEMPLATES_DIR / fname)) for key, fname in DEAD_TEMPLATES.items()}
    full = {key: to_grayscale(Image.open(TEMPLATES_DIR / fname)) for key, fname in FULL_KING_TEMPLATES.items()}
    return dead, full


def load_elixir_templates():
    return {i: to_grayscale(Image.open(TEMPLATES_DIR / f"elixir_{i}.jpg")) for i in range(11)}


# ----- PERCEPTION COMPONENTS -----

def run_yolo(arena_image, model):
    """Run YOLO on arena image -> returns list of (x1, y1, x2, y2, class_id, conf)"""
    results = model(arena_image, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append((x1, y1, x2, y2, class_id, conf))

    return detections


def build_troop_tensor(detections):
    """Convert detections to (classes, rows, cols) binary troop location tensor (for policy network input)"""
    tensor = np.zeros((NUM_CLASSES, GRID_ROWS, GRID_COLS), dtype=np.float32)
    x_min, y_min, x_max, y_max = ARENA_TILE_BOUNDS
    tile_w = (x_max-x_min) / GRID_COLS
    tile_h = (y_max-y_min) / GRID_ROWS

    for x1, y1, x2, y2, class_id, _ in detections:
        troop_x = (x1+x2) / 2
        troop_y = (y1+y2) / 2
        col = int(troop_x / tile_w)
        row = int(troop_y / tile_h)
        col = max(0, min(GRID_COLS-1, col))
        row = max(0, min(GRID_ROWS-1, row))
        tensor[class_id, row, col] = 1.0

    return tensor


def best_ssim_match(crop, templates):
    """Return key of template that is most similar to crop (via SSIM)"""
    crop_gray = to_grayscale(crop)
    best_score = -1.0
    best_key = next(iter(templates))
    for key, template in templates.items():
        score = ssim(crop_gray, template)
        if score > best_score:
            best_score = score
            best_key = key

    return best_key


def identify_cards(crops):
    """Identify cards each slot 1-4 and return flattened one-hot encoding (32,)"""
    one_hots = []
    for slot in ("card_1", "card_2", "card_3", "card_4"):
        card_name = best_ssim_match(crops[slot], CARD_TEMPLATES)
        idx = TROOP_NAMES.index(card_name)
        vec = np.zeros(8, dtype=np.float32)
        vec[idx] = 1.0
        one_hots.append(vec)

    return np.concatenate(one_hots)


def threshold_for_ocr(image, rgb_center, tol):
    """Convert image to a binary using our custom thresholding for easy OCRing -> goal is numbers = black, everything else = white"""
    if isinstance(image, np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.array(image.convert("RGB"))

    r, g, b = rgb_center
    mask = (
        (rgb[:, :, 0] >= r-tol) & (rgb[:, :, 0] <= r+tol) &
        (rgb[:, :, 1] >= g-tol) & (rgb[:, :, 1] <= g+tol) &
        (rgb[:, :, 2] >= b-tol) & (rgb[:, :, 2] <= b+tol)
    )
    binary = np.where(mask, np.uint8(0), np.uint8(255))

    # Upscale if tiny
    h, w = binary.shape[:2]
    scale = max(1, 120 // min(h, w))
    if scale > 1:
        binary = cv2.resize(binary, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    return binary


def ocr_digits(img, rgb_center, rgb_tolerance):
    """OCR to detect tower HP value digits"""
    binary = threshold_for_ocr(img, rgb_center, rgb_tolerance)
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255) # not too close to edges
    results = OCR_READER.readtext(binary, allowlist='0123456789', detail=0, min_size=5)
    digits = re.sub(r"\D", "", ''.join(results)) # ONLY NUMBERS PLEZ

    return int(digits) if digits else None


def detect_tower_hp(crops):
    """Returns normalized [0,1] tower HP array (6,)"""
    hp_values = np.zeros(6, dtype=np.float32)
    for i, key in enumerate(TOWER_KEYS):
        crop_gray = to_grayscale(crops[key])
        hp_max = TOWER_HP_MAX[i]

        if key in DEAD_TEMPLATES and ssim(crop_gray, TOWER_DEAD_TEMPLATES[key]) >= SSIM_THRESHOLD:
            hp_values[i] = 0.0
            continue

        if key in FULL_KING_TEMPLATES and ssim(crop_gray, TOWER_FULL_TEMPLATES[key]) >= SSIM_THRESHOLD:
            hp_values[i] = 1.0
            continue

        rgb_center, rgb_tol = TOWER_OCR_COLORS[key]
        raw = ocr_digits(crops[key], rgb_center, rgb_tol)
        hp = raw if raw is not None else 0
        hp_values[i] = float(np.clip(hp/hp_max, 0.0, 1.0))

    return hp_values


# ----- PERCEPTION PIPELINE -----

def perceive(crops, yolo_path=YOLO_PATH):
    """Convert image crops into the current screen name and inputs for policy network (screen_name, troop_tensor, flat_vector)"""
    # Caching stuff
    global YOLO_LOADED, YOLO_MODEL, CARD_TEMPLATES, ELIXIR_TEMPLATES, TOWER_DEAD_TEMPLATES, TOWER_FULL_TEMPLATES, SCREEN_TEMPLATES, OCR_READER
    if SCREEN_TEMPLATES is None:
        SCREEN_TEMPLATES = load_screen_templates()
    if not YOLO_LOADED and yolo_path is not None:
        YOLO_MODEL = YOLO(yolo_path)
        YOLO_LOADED = True
    if CARD_TEMPLATES is None:
        CARD_TEMPLATES = load_card_templates()
    if ELIXIR_TEMPLATES is None:
        ELIXIR_TEMPLATES = load_elixir_templates()
    if TOWER_DEAD_TEMPLATES is None:
        TOWER_DEAD_TEMPLATES, TOWER_FULL_TEMPLATES = load_tower_templates()
    if OCR_READER is None:
        OCR_READER = easyocr.Reader(['en'], gpu=False) # no mps supp

    screen_name = best_ssim_match(crops["screen"], SCREEN_TEMPLATES)

    if screen_name != "match":
        return screen_name, None, None

    detections = run_yolo(crops["arena"], YOLO_MODEL)
    troop_tensor = build_troop_tensor(detections)

    cards_onehots_flat = identify_cards(crops)
    tower_hp = detect_tower_hp(crops)
    elixir = np.array([int(best_ssim_match(crops["elixir"], ELIXIR_TEMPLATES)) / 10], dtype=np.float32)

    flat_vector = np.concatenate([cards_onehots_flat, tower_hp, elixir])

    return screen_name, troop_tensor, flat_vector


# ----- TESTING -----

if __name__ == "__main__":
    path = "screenshots/remaining_dead_towers.jpg"
    print(f"Perceiving: {path}")

    crops = process_screenshot(path)
    screen_name, troop_tensor, flat_vector = perceive(crops)

    print(f"Screen: {screen_name}")

    if screen_name != "match":
        print("Not in a match -> no game state.")
    else:
        cards_onehots = flat_vector[:32].reshape(4, 8)
        tower_hp = flat_vector[32:38]
        elixir = flat_vector[38]

        print("\nCards in hand:")
        for slot, vector in enumerate(cards_onehots, 1):
            card_idx = int(np.argmax(vector))
            print(f"Slot {slot}: {TROOP_NAMES[card_idx]}")

        print("\nTower HP (normalized):")
        for key, val in zip(TOWER_KEYS, tower_hp):
            print(f"{key}: {val}")

        print(f"\nElixir (normalized): {elixir}")

        troop_counts = troop_tensor.sum(axis=(1, 2))
        ally_names = [f"ally_{name}" for name in TROOP_NAMES]
        enemy_names = [f"enemy_{name}" for name in TROOP_NAMES]
        all_names = ally_names + enemy_names
        detected = [(name, int(count)) for name, count in zip(all_names, troop_counts) if count > 0]
        print("\nDetected troops:")
        if detected:
            for name, count in detected:
                print(f"{name}: {count}")
        else:
            print("N/A")