import time
import torch
import warnings

from capture_images import take_screenshot, process_screenshot
from perception import perceive
from execute_action import execute_action
from policy_network import load_policy

warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning) # get rid of annoying warning


STEP_INTERVAL = 1.0
WAIT_ACTION = 32

DEVICE = "mps"


def action_label(action):
    if action == WAIT_ACTION:
        return "Action 32 (wait)"
    card_slot = (action // 8) + 1
    tile_idx = action % 8
    return f"Action {action} (card {card_slot}, tile {tile_idx})"


if __name__ == "__main__":
    policy_path = "policy_270.pt"
    model = load_policy(policy_path, device=DEVICE)
    print(f"Loaded policy from {policy_path}, starting in 5 seconds...")
    time.sleep(5)

    while True:
        ss = take_screenshot()
        crops = process_screenshot(ss)
        screen_name, troop_tensor, flat_vector = perceive(crops)

        if screen_name != "match":
            print(f"NOT IN MATCH: {screen_name}")
            time.sleep(STEP_INTERVAL)
            continue

        troop_tensor = torch.from_numpy(troop_tensor).unsqueeze(0).to(DEVICE)
        flat_vector = torch.from_numpy(flat_vector).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q = model(troop_tensor, flat_vector).squeeze(0)

        action = int(q.argmax().item())
        q_list = [round(v, 1) for v in q.tolist()]

        print(f"Qs: {q_list}")
        print(action_label(action))
        execute_action(action)

        time.sleep(STEP_INTERVAL)