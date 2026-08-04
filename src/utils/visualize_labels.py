import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from src.paths import DATA_DIR


# ----- Config -----


DATASET_DIR = str(DATA_DIR / "synthetic_dataset")

TROOP_NAMES = ["archer", "giant", "goblin", "knight", "mekka", "minion", "musketeer", "spoblin"]
SIDES = ["ally", "enemy"]
CLASS_NAMES = [f"{troop}_{side}" for side in SIDES for troop in TROOP_NAMES] # Class ID -> sprite folder


# ----------


def load_bboxes(label_path, image_w, image_h):
    """Converts all YOLO labels to bbox corner coords"""
    bboxes = []
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, center_x, center_y, bbox_w, bbox_h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((center_x - (bbox_w/2)) * image_w)
            x2 = int((center_x + (bbox_w/2)) * image_w)
            y1 = int((center_y - (bbox_h/2)) * image_h)
            y2 = int((center_y + (bbox_h/2)) * image_h)
            bboxes.append((class_id, x1, x2, y1, y2))
    
    return bboxes


def main():
    image_dir = os.path.join(DATASET_DIR, "images")
    label_dir = os.path.join(DATASET_DIR, "labels")
    all_images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))

    print("Any key to go next, Q to quit")

    for filename in all_images:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)
        image_h, image_w = image.shape[:2]

        for class_id, x1, x2, y1, y2 in load_bboxes(label_path, image_w, image_h):
            color = (0, 0, 255)
            label = CLASS_NAMES[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2) # bbox
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-text_h-4), (x1+text_w, y1), color, -1) # filled background for label
            cv2.putText(image, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # label

        cv2.imshow(filename, image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
