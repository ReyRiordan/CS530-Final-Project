import time
import numpy as np
import pyautogui

from capture_images import take_screenshot, process_screenshot
from perception import perceive
from execute_action import execute_action

STEP_INTERVAL = 1.0

# flat_vector indices (see perception.py, TOWER_KEYS order)
ALLY_HP = slice(32, 35)
ENEMY_HP = slice(35, 38)
ELIXIR_IDX = 38

PRINCESS_REWARD_SCALE = 10.0
KING_REWARD_SCALE = 20.0
PRINCESS_DESTROY_BONUS = 5.0
MAX_HP_DELTA = 0.5  # max believable tower HP loss percentage per step -> try to counteract OCR failures

# Elixir costs indexed by TROOP_NAMES = ["archer", "giant", "goblin", "knight", "mekka", "minion", "musketeer", "spoblin"]
CARD_COSTS = [3, 5, 2, 3, 4, 3, 4, 2]

INVALID_PLACEMENT_PENALTY = -0.5
ELIXIR_LEAK_PENALTY = -0.2

# Click coords for menu navigation
CLICK_MAP = {
    "end": (735, 874), # click OK after match ends
    "main": (939, 166), # click 3 bars
    "3bars": (788, 360), # click training match
    "startconfirm": (826, 581), # click ok to confirm training match
}

MAX_NAV_RETRIES = 30
NAV_WAIT_INTERVAL = 2.0


def compute_reward(prev_ally_hp, prev_enemy_hp, cur_ally_hp, cur_enemy_hp):
    """Reward based on tower HP change + destroy bonus -> positive for enemy damage, negative for ally damage"""
    enemy_princess_delta = float(np.sum(np.clip(prev_enemy_hp[:2] - cur_enemy_hp[:2], 0, MAX_HP_DELTA)))
    ally_princess_delta = float(np.sum(np.clip(prev_ally_hp[:2]  - cur_ally_hp[:2], 0, MAX_HP_DELTA)))
    enemy_king_delta = float(np.clip(prev_enemy_hp[2] - cur_enemy_hp[2], 0, MAX_HP_DELTA))
    ally_king_delta = float(np.clip(prev_ally_hp[2]  - cur_ally_hp[2], 0, MAX_HP_DELTA))
    reward = (
        (enemy_princess_delta - ally_princess_delta) * PRINCESS_REWARD_SCALE +
        (enemy_king_delta - ally_king_delta) * KING_REWARD_SCALE
    )
    for i in range(2):
        if prev_enemy_hp[i] > 0 and cur_enemy_hp[i] == 0:
            reward += PRINCESS_DESTROY_BONUS
        if prev_ally_hp[i] > 0 and cur_ally_hp[i] == 0:
            reward -= PRINCESS_DESTROY_BONUS
    return reward


def action_penalty(action, flat_vector):
    """Penalty for invalid card placement (not enough elixir) or leaking elixir (waiting at 10)"""
    penalty = 0
    elixir = flat_vector[ELIXIR_IDX] * 10
    if action < 32:
        slot = action // 8
        troop_idx = int(np.argmax(flat_vector[slot*8 : (slot+1)*8]))
        if elixir < CARD_COSTS[troop_idx]:
            penalty += INVALID_PLACEMENT_PENALTY
    else:
        if flat_vector[ELIXIR_IDX] >= 1.0:
            penalty += ELIXIR_LEAK_PENALTY
    return penalty


class ClashRoyaleEnv:
    """
    OpenAI Gymnasium format-ish wrapper for us
    State: troop_tensor (16,32,18), flat_vector (39,)
    Actions: (33,) -> 0-31 place card, 32 wait
    """

    def __init__(self):
        self.prev_ally_hp = None # float (3,): [ally_left, ally_right, ally_king]
        self.prev_enemy_hp = None # float (3,): [enemy_left, enemy_right, enemy_king]
        self.prev_flat_vector = None

    def reset(self):
        """Navigate to a fresh match and return the initial state"""
        _, troop_tensor, flat_vector = self.navigate_to_match()
        self.prev_ally_hp = flat_vector[ALLY_HP].copy()
        self.prev_enemy_hp = flat_vector[ENEMY_HP].copy()
        self.prev_flat_vector = flat_vector.copy()
        return troop_tensor, flat_vector

    def step(self, action):
        """Execute action, wait interval, get new state, compute reward -> returns state, reward, done, screen name"""
        execute_action(action)
        time.sleep(STEP_INTERVAL)
        screen_name, troop_tensor, flat_vector = self.get_game_state()

        if screen_name != "match":
            reward = 0.0
            self.prev_ally_hp = None
            self.prev_enemy_hp = None
            self.prev_flat_vector = None
            return self.zero_state(), reward, True, screen_name

        reward = compute_reward(self.prev_ally_hp, self.prev_enemy_hp, flat_vector[ALLY_HP], flat_vector[ENEMY_HP])
        reward += action_penalty(action, self.prev_flat_vector)

        self.prev_ally_hp = flat_vector[ALLY_HP].copy()
        self.prev_enemy_hp = flat_vector[ENEMY_HP].copy()
        self.prev_flat_vector = flat_vector.copy()

        return (troop_tensor, flat_vector), reward, False, screen_name

    # ----------

    def get_game_state(self):
        ss = take_screenshot()
        crops = process_screenshot(ss)
        return perceive(crops)

    def navigate_to_match(self):
        """Auto navigate to start new training match, returns perception result if in match"""
        for r in range(MAX_NAV_RETRIES):
            screen_name, troop_tensor, flat_vector = self.get_game_state()
            if screen_name == "match":
                return screen_name, troop_tensor, flat_vector
            
            coords = CLICK_MAP.get(screen_name)
            if coords:
                pyautogui.click(*coords)
            time.sleep(NAV_WAIT_INTERVAL)

        raise RuntimeError(f"Could not reach match screen after {MAX_NAV_RETRIES} retries")

    def zero_state(self):
        return np.zeros((16, 32, 18), dtype=np.float32), np.zeros(39, dtype=np.float32)