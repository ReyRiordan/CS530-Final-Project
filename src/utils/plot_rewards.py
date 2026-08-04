import matplotlib.pyplot as plt
import pandas as pd

from src.paths import OUTPUTS_DIR

REWARDS_FILE = OUTPUTS_DIR / "ep_rewards.txt"
EPISODE_START = 50
MOVING_AVG_N = 20

with open(REWARDS_FILE) as f:
    rewards = [float(line.strip().split("=")[1]) for line in f if line.strip()]

episodes = list(range(EPISODE_START, EPISODE_START + len(rewards)))
moving_avg = pd.Series(rewards).rolling(window=MOVING_AVG_N, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(episodes, rewards, alpha=0.4, linewidth=1, label="Episodic reward")
plt.plot(episodes, moving_avg, linewidth=2, label=f"Moving average (N={MOVING_AVG_N})")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episodic Rewards during RL Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "reward_plot.png", dpi=150)
plt.show()
