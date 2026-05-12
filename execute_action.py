import pyautogui
from PIL import Image, ImageDraw, ImageFont


# ----- CONFIG -----

# Weird pixel coord scaling issue due to macOS Retina
DISPLAY_SCALE = 2

# Coords: top left of arena
ARENA_TOP_LEFT = (1050, 285)

ARENA_PIXEL_W = 840
ARENA_PIXEL_H = 1215

TILE_W = ARENA_PIXEL_W / 18
TILE_H = ARENA_PIXEL_H / 32

# (row, col) coords for the 8 placement tiles
PLACEMENT_TILES = [
    (17, 3), # 0: left bridge
    (20, 7), # 1: left center
    (24, 5), # 2: left tower
    (30, 7), # 3: left king
    (17, 14), # right bridge
    (20, 10), # right center
    (24, 12), # right tower
    (30, 10), # right king
]


# ----------


def tile_to_screen(row, col):
    """Calculate tile coordinates on screen"""
    x = int(ARENA_TOP_LEFT[0] + (col+0.5) * TILE_W)
    y = int(ARENA_TOP_LEFT[1] + (row+0.5) * TILE_H)
    return (x, y)


def place_card(card_slot, tile_idx):
    row, col = PLACEMENT_TILES[tile_idx]
    click_x, click_y = tile_to_screen(row, col)
    pyautogui.press(str(card_slot))
    pyautogui.click(click_x // DISPLAY_SCALE, click_y // DISPLAY_SCALE)


def execute_action(action: int):
    """Execute action: 0-7 = place card 1, ..., 32 = wait"""
    if not (0 <= action < 33):
        raise ValueError(f"Invalid action: {action}")
    if action != 32: # if not "wait"
        card_slot = (action // 8) + 1 # 1-indexed (1–4)
        tile_idx  = action % 8 # 0-indexed (0–7)
        place_card(card_slot, tile_idx)


if __name__ == "__main__":
    image = Image.open("screenshots/arena_init_empty.jpg").copy()
    draw = ImageDraw.Draw(image)

    origin_x, origin_y = ARENA_TOP_LEFT

    # grid lines (32 x 18)
    for col in range(19):
        x = origin_x + col * TILE_W
        draw.line([(x, origin_y), (x, origin_y+ARENA_PIXEL_H)], fill="red", width=1)
    for row in range(33):
        y = origin_y + row * TILE_H
        draw.line([(origin_x, y), (origin_x+ARENA_PIXEL_W, y)], fill="red", width=1)

    # Dot + label for each of 8 action placement locations
    for i, (tile_row, tile_col) in enumerate(PLACEMENT_TILES):
        dot_x, dot_y = tile_to_screen(tile_row, tile_col)
        dot_radius = 10
        draw.ellipse([(dot_x-dot_radius, dot_y-dot_radius), (dot_x+dot_radius, dot_y+dot_radius)], fill="red", outline="white", width=2)
        draw.text((dot_x+dot_radius+2, dot_y-8), str(i), fill="white", font=ImageFont.load_default(size=32))

    # Crop to arena view only
    arena_crop = image.crop((origin_x, origin_y, origin_x+ARENA_PIXEL_W, origin_y+ARENA_PIXEL_H))

    arena_crop.save("screenshots/arena_grid_overlay.jpg", quality=90)
    print(f"Saved to screenshots/arena_grid_overlay.jpg")
