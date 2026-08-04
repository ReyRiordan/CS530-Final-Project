# CS530-Final-Project

## Codebase

Run scripts as modules from the repo root, e.g. `python -m src.training.train_rl` or `python -m src.agent.play_policy`.

### Code

`src/vision/` -> perception + synthetic data pipeline:

[synthetic_generation.py](src/vision/synthetic_generation.py) -> generate N synthetic data examples

[train_yolo.ipynb](src/vision/train_yolo.ipynb) -> code to fine-tune YOLO26 with synthetic data (on Google Colab free GPU)

[validate_yolo.py](src/vision/validate_yolo.py) -> quick yoink of built-in YOLO validation

[capture_images.py](src/vision/capture_images.py) -> take screenshot and crop/segment into 13 relevant regions

[perception.py](src/vision/perception.py) -> take cropped image input from capture_images.py, process with YOLO/SSIM/OCR to exact numerical state representations that policy network uses as input

`src/agent/` -> the deployed agent (perceive -> decide -> act):

[policy_network.py](src/agent/policy_network.py) -> CNN/DQN that takes in game state and Q estimates for 33 actions

[execute_action.py](src/agent/execute_action.py) -> executes an action using PyAutoGUI

[play_policy.py](src/agent/play_policy.py) -> play matches with agent policy and allat

`src/training/` -> RL training:

[environment.py](src/training/environment.py) -> Gym environment wrapper for RL training, automatic menu navigation to start new match

[replay_buffer.py](src/training/replay_buffer.py) -> replay buffer for RL training

[train_rl.py](src/training/train_rl.py) -> RL training, load recorded human data with undersampling

`src/utils/` -> utilities:

[record_data.py](src/utils/record_data.py) -> Records state + actions into data/human_data/ while human is playing on emulator

[visualize_labels.py](src/utils/visualize_labels.py) -> visualize bounding boxes + labels for synthetically generated examples

[extract_policy.py](src/utils/extract_policy.py) -> extract standalone policy weights from a training checkpoint

[plot_rewards.py](src/utils/plot_rewards.py) -> plot episodic reward curve from training log

### Files

[models/](models/) -> trained weights: YOLO troop detector, best policy, and RL training [checkpoints/](models/checkpoints/)

[data/sprites/](data/sprites/) -> all sprites for 16 troop classes (100-200 transparent pngs each), used for synthetic data generation

[data/synthetic_dataset/](data/synthetic_dataset/) -> dataset config, full dataset (3k examples) is too big for repo

[src/vision/templates/](src/vision/templates/) -> images of all cards + tower/arena states used by the perception pipeline (SSIM matching)

[data/human_data/](data/human_data/) -> recorded human data, states/ and actions/

[outputs/screenshots/](outputs/screenshots/) -> screenshots used for testing, cropping, etc

[outputs/runs/](outputs/runs/) -> auto-generated YOLO validation stats

[outputs/recording.gif](outputs/recording.gif) -> demo recording of the agent playing

[Principles of AI - Final Report.pdf](Principles%20of%20AI%20-%20Final%20Report.pdf) -> final report

## Plan

### Platform / Environment

Running the game:
- MacBook Air M2 running Mumu Player (Android Emulator)
- New Clash Royale Account, played just enough to unlock necessary arena 1 cards
- Playing against arena 1 training camp built-in bot opponent (Trainer Jonas/George)
- Can play real battles but have to be careful to not rank up to next arena

Environment observations: capture screen (1 per sec) with python mss?
Crop to only emulated phone screen, then crop/segment into the following:
- Arena (where troops are)
- Cropped part of each of current hand of cards (4 slots)
- Cropped part of just current exixir value
- Cropped part of ally/enemy tower HP (6 towers total)

Possible states:
- 8 troops
- Cards in hand: 4 out of 8 possible cards that are in deck
- Elixir: 0-10
- Tower HP: 0-1512 for princess towers (2 on each side), 2568 for king towers (1 on each side), need way to detect when princess tower is down (convert to 0 HP) or king tower is full health (convert to 2568 HP) because HP value is not visible as numbers in these cases
- NOTE 1: ignoring troop HP as it adds too much complexity, ignoring troop and tower levels since we standardized them
- NOTE 2: ignoring spells/buildings due to no sprites, using deck of 8 troops to avoid this mess + halve-ish action space (troops can usually only be placed on ally side)

### Perception Pipeline

Simple computer vision (cards in hand, elixir, tower HP):
- Cards in hand: take card slot crops, compare against the 8 known cropped view of cards in deck using greyscale conversion + SSIM, take best match (handles grayscale effect when not enough elixir)
- Tower HP: take crop of numbers, use custom thresholding to convert to binary (black digits on white background), OCR with EasyOCR
- Elixir: take crop of elixir value, compare against 0-10 elixir value templates, take best match

YOLO network (troop detection):
- Pretrained YOLO: Ultralytics YOLO26 -> https://docs.ultralytics.com/models/yolo26/
- 16 classes (8 ally troops, 8 enemy troops)

Synthetic data generation (for troop detection YOLO training):
- Python script that picks random number of troops present (1-10?), picks random troops, picks random sprites for each troop, then pastes transparent sprites in random valid locations on screenshot of empty arena 1
- Same script also generates the ground truth class labels and bounding boxes at the same time (chosen troops, bounding boxes as border of sprite image)
- Can add slight random noise/scaling as well to troop sprites -> improved robusticity
- Generate at least a few thousand examples

Training troop detection YOLO:
- Split synthetically generated dataset into training and validation sets
- Fine-tune YOLO26 for N epochs
- Stop when out of time or validation accuracy plateaus
- Freeze backbone (train only head/neck first), might use minor data augmentation
- https://docs.ultralytics.com/guides/finetuning-guide/#freezing-layers
- https://docs.ultralytics.com/guides/custom-trainer/
- https://docs.ultralytics.com/guides/yolo-data-augmentation/#hue-adjustment-hsv_h

### State Representations

Environment: construct state tensor (C, H, W) where H=32 and W=18 (arena tiling dimensions) and C=16+ feature channels
- Channel 0-7: ally troops (1 if that troop at that tile, 0 if not)
- Channel 8-15: enemy troops
- Elixir + tower HP: each normalized to [0,1]
- Cards in hand: one-hot encoded vector x 4

Actions: discrete action space of 33 actions (whether to play card, which card to play, where to play card)
- Discrete choice of playing each card in hand 1-4 in one of 8 tiles in arena, or just "wait" without playing card -> (4 cards x 8 tiles) + 1 wait = 33 total actions
- NOTE 1: 224 valid tiles, but shrinking action space to 8 key tiles for manageable action space + macro placement matters more at beginner level
- NOTE 2: more valid tiles open up to place troops after taking an enemy tower, but we are ignoring this for simplicity

### Agent Policy

Policy Network Architecture:
- Use CNN as the deep Q estimation network
- Convolutional layers input: troop locations state tensor (16x32x18 -> 16 binary 32x18 arrays that represent if/where that class/troop is present in terms of arena tiles)
- Dense layers input: flatten+concatenation of feature vector from convolutional layers + cards in hand (4 one-hot vectors, each size 8) + normalized tower HP values (6 total) + normalized elixir value
- CNN output: Q estimation for each action (size 33), max determines which action to take

Reinforcement Learning + Human Behavior Bootstrapping:
- Use double DQN (CNN) with replay buffer, pre-fill buffer with human data
- Python script that records data while human plays on emulator: record environment state every second, record actions taken (attach actions to nearest state for dataset)
- Human plays N matches against arena 1 training camp bot (try to play predictably, use similar counters and build up pushes in similar ways)
- Undersample "wait" frames if excessively overrepresented in data
- Positive rewards: deal tower damage, take opponent tower, win game
- Negative rewards: take tower damage, lose tower, lose game, "leak" elixir, invalid action (not enough elixir to place card)
- Auto-play training games to RL train