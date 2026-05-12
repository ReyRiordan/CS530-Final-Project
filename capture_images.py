import os
from PIL import Image, ImageGrab

# ----- CROP CONFIGS: (left, top, right, bottom) -----

# Crop phone region only first (separate config in case this sht moves)
PHONE_REGION = (970, 0, 1970, 1912)

# Now crop within phone region
# ARENA_CROP = (0, 140, 1000, 1570)
SCREEN_CROP = (0, 1100, 1000, 1912) # to detect what screen we on (for auto replaying etc)
ARENA_CROP = (80, 285, 920, 1500)
CARD_1_CROP = (225, 1612, 395, 1827)
CARD_2_CROP = (412, 1612, 582, 1827)
CARD_3_CROP = (599, 1612, 769, 1827)
CARD_4_CROP = (786, 1612, 956, 1827)
ELIXIR_CROP = (265, 1820, 325, 1880)
TOWER_HP_ALLY_KING_CROP = (465, 1475, 560, 1520)
TOWER_HP_ALLY_LEFT_CROP = (200, 1240, 265, 1270)
TOWER_HP_ALLY_RIGHT_CROP = (720, 1240, 785, 1270)
TOWER_HP_ENEMY_KING_CROP = (465, 162, 560, 207)
TOWER_HP_ENEMY_LEFT_CROP = (200, 377, 265, 407)
TOWER_HP_ENEMY_RIGHT_CROP = (720, 377, 785, 407)


# ----------


def take_screenshot():
    """Take screenshot, return as a PIL Image"""
    return ImageGrab.grab()


def process_screenshot(ss):
    """Process full screenshot into 12 crops needed for perception pipeline"""
    # If path, process from screenshots/ (only for my testing purposes)
    if isinstance(ss, str):
        ss = Image.open(ss)

    phone = ss.crop(PHONE_REGION)

    crops = {
        "screen": phone.crop(SCREEN_CROP),
        "arena": phone.crop(ARENA_CROP),
        "card_1": phone.crop(CARD_1_CROP),
        "card_2": phone.crop(CARD_2_CROP),
        "card_3": phone.crop(CARD_3_CROP),
        "card_4": phone.crop(CARD_4_CROP),
        "elixir": phone.crop(ELIXIR_CROP),
        "tower_hp_enemy_left": phone.crop(TOWER_HP_ENEMY_LEFT_CROP),
        "tower_hp_enemy_right": phone.crop(TOWER_HP_ENEMY_RIGHT_CROP),
        "tower_hp_enemy_king": phone.crop(TOWER_HP_ENEMY_KING_CROP),
        "tower_hp_ally_left": phone.crop(TOWER_HP_ALLY_LEFT_CROP),
        "tower_hp_ally_right": phone.crop(TOWER_HP_ALLY_RIGHT_CROP),
        "tower_hp_ally_king": phone.crop(TOWER_HP_ALLY_KING_CROP),
    }

    return crops


# ----------


if __name__ == "__main__":
    screenshot_path = "screenshots/test1.jpg"
    print(f"Processing ss from path: {screenshot_path}")

    crops = process_screenshot(screenshot_path)

    os.makedirs("crops", exist_ok=True)
    for name, crop in crops.items():
        out_path = os.path.join("crops", f"{name}.jpg")
        crop.save(out_path)
    print(f"Saved crops yippee")