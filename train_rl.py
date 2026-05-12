import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import warnings

from replay_buffer import ReplayBuffer
from environment import ClashRoyaleEnv, ALLY_HP, ENEMY_HP, compute_reward, action_penalty
from policy_network import PolicyNetwork, select_action

warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning) # get rid of annoying warning


# ----- CONFIG -----

DEVICE = "mps"

BUFFER_CAPACITY = 10000
MIN_BUFFER = 500
WAIT_KEEP_PROB = 0.75 # ~72% "wait" in human data -> only mild undersampling

BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0001
TARGET_UPDATE_FREQ = 500 # gradient steps between updates
GRADIENT_CLIP = 10.0 # just prevent really bad spikes

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98

HUMAN_STATES_DIR = "human_data/states"
HUMAN_ACTIONS_DIR = "human_data/actions"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 10 # episodes


# ----------

def load_human_data():
    """Load human data as (state, action, reward, next_state, done) transitions, wait actions undersampled"""
    state_paths = sorted(glob.glob(os.path.join(HUMAN_STATES_DIR, "?????.npz")))
    action_paths = sorted(glob.glob(os.path.join(HUMAN_ACTIONS_DIR, "?????.npy")))
    if not state_paths:
        return []

    states = {}
    actions = {}

    for state_path in state_paths:
        state_id = int(os.path.splitext(os.path.basename(state_path))[0])
        data = np.load(state_path)
        states[state_id] = (data["troop_tensor"].astype(np.float32), data["flat_vector"].astype(np.float32))

    for action_path in action_paths:
        state_id = int(os.path.splitext(os.path.basename(action_path))[0])
        actions[state_id] = int(np.load(action_path))

    sorted_ids = sorted(states.keys())
    transitions = []

    for i, state_id in enumerate(sorted_ids):
        if state_id not in actions:
            continue

        action = actions[state_id]
        if action == 32 and random.random() > WAIT_KEEP_PROB:
            continue

        troop_tensor, flat_vector = states[state_id]

        next_state_id = sorted_ids[i+1] if i+1 < len(sorted_ids) else None
        consecutive = next_state_id is not None and next_state_id == state_id+1

        if consecutive:
            next_troop_tensor, next_flat_vector = states[next_state_id]
            done = False
            reward = compute_reward(flat_vector[ALLY_HP], flat_vector[ENEMY_HP], next_flat_vector[ALLY_HP], next_flat_vector[ENEMY_HP]) + action_penalty(action, flat_vector)
        else:
            next_troop_tensor = np.zeros_like(troop_tensor)
            next_flat_vector = np.zeros_like(flat_vector)
            done = True
            reward = 0.0

        transitions.append((troop_tensor, flat_vector, action, reward, next_troop_tensor, next_flat_vector, done))

    return transitions


def gradient_step(online_net, target_net, optimizer, buffer):
    troop_tensor, flat_vector, actions, rewards, next_troop_tensor, next_flat_vector, dones = buffer.sample(BATCH_SIZE)

    troop_tensor = torch.from_numpy(troop_tensor).to(DEVICE)
    flat_vector = torch.from_numpy(flat_vector).to(DEVICE)
    actions = torch.from_numpy(actions).to(DEVICE)
    rewards = torch.from_numpy(rewards).to(DEVICE)
    next_troop_tensor = torch.from_numpy(next_troop_tensor).to(DEVICE)
    next_flat_vector = torch.from_numpy(next_flat_vector).to(DEVICE)
    dones = torch.from_numpy(dones).to(DEVICE)

    # double DQN -> online picks action, target evaluates value
    with torch.no_grad():
        next_actions = online_net(next_troop_tensor, next_flat_vector).argmax(dim=1, keepdim=True)
        next_q = target_net(next_troop_tensor, next_flat_vector).gather(1, next_actions).squeeze(1)
        targets = rewards + GAMMA * next_q * (1.0-dones)

    q_vals = online_net(troop_tensor, flat_vector).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = nn.functional.huber_loss(q_vals, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), GRADIENT_CLIP)
    optimizer.step()

    return loss.item()


def train(num_episodes=500, resume_path=None):
    """Pre-fills buffer with human data and trains for specified episodes (just leave computer running), can also resume from checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    online_net = PolicyNetwork().to(DEVICE)
    target_net = PolicyNetwork().to(DEVICE)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    start_ep = 0
    global_step = 0
    epsilon = EPSILON_START

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        online_net.load_state_dict(checkpoint["online_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_ep = checkpoint["episode"]
        global_step = checkpoint["step"]
        epsilon = checkpoint["epsilon"]
        print(f"Resumed from episode {start_ep}, step {global_step}")

    human_transitions = load_human_data()
    for transition in human_transitions:
        buffer.add(*transition)
    print(f"Human data: loaded {len(human_transitions)} transitions -> {len(buffer)} in buffer after wait undersampling")

    env = ClashRoyaleEnv()

    for ep in range(start_ep, num_episodes):
        troop_tensor, flat_vector = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            action = select_action(online_net, troop_tensor, flat_vector, epsilon=epsilon, device=DEVICE)
            (next_troop_tensor, next_flat_vector), reward, done, screen_name = env.step(action)

            buffer.add(troop_tensor, flat_vector, action, reward, next_troop_tensor, next_flat_vector, done)

            if len(buffer) >= MIN_BUFFER:
                loss = gradient_step(online_net, target_net, optimizer, buffer)
                global_step += 1
                if global_step % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    print("UPDATED TARGET NETWORK")

            troop_tensor = next_troop_tensor
            flat_vector = next_flat_vector
            ep_reward += reward
            ep_steps += 1

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        print(f"Ep {ep} | ep_steps={ep_steps} | ep_reward={ep_reward} | epsilon={epsilon} | buffer_size={len(buffer)} | global_step={global_step}")

        # TODO: prob need buffer content for a real resume but fk it for now
        if (ep+1) % CHECKPOINT_FREQ == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ep{ep+1:05d}.pt")
            torch.save({
                "online_net": online_net.state_dict(),
                "target_net": target_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep+1,
                "step": global_step,
                "epsilon": epsilon,
            }, path)
            print(f"SAVED CHECKPOINT: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(num_episodes=args.episodes, resume_path=args.resume)