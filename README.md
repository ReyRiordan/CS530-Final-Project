# CS530-Final-Project

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
- Cards in hand: take card slot crops, compare against the 8 known cropped view of cards in deck using greyscale conversion + SSIM, take best match (handles visual effect when not enough elixir)
- Elixir and Tower HP: take crop of numbers, use basic thresholding to convert to black/white, OCR it (PyTesseract?)

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

Reinforcement Learning + Human Behavior Bootstrapping:
- Use double DQN (CNN) with Prioritized Experience Replay (PER) buffer, pre-fill buffer with human data
- Python script that records data while human plays on emulator: record environment state every second, record actions taken (attach actions to nearest state for dataset)
- Human plays N matches against arena 1 training camp bot (try to play predictably, use similar counters and build up pushes in similar ways)
- Undersample "wait" frames if excessively overrepresented in data
- Positive rewards: deal tower damage, take opponent tower, win game
- Negative rewards: take tower damage, lose tower, lose game, "leak" elixir, invalid action (not enough elixir to place card)
- Auto-play training games to RL train

CNN Architecture:
- Use CNN as the deep Q estimation network
- Convolutional layers input: troop locations state tensor (16x32x18 -> 16 binary 32x18 arrays that represent if/where that class/troop is present in terms of arena tiles)
- Dense layers input: flatten+concatenation of feature vector from convolutional layers + cards in hand (4 one-hot vectors, each size 8) + normalized tower HP values (6 total) + normalized elixir value
- CNN output: Q estimation for each action (size 33), max determines which action to take

## Codebase

### Code

capture_images.py -> take screenshot and crop/segment into 12 relevant regions
synthetic_generation.py -> generate N synthetic data examples
visualize_labels.py -> visualize bounding boxes + labels for synthetically generated examples
train_yolo.py -> code to fine-tune YOLO26 with synthetic data
record_data.py -> UNFINISHED, supposed to record state + actions while human is playing on emulator
perception.py -> take cropped image input from capture_images.py, process with YOLO/SSIM/OCR to exact numerical state representations that CNN uses as input

### Files

sprites/ -> all sprites for 16 troop classes (100-200 transparent pngs each), used for synthetic data generation
synthetic_dataset/ -> all synthetically generated data to fine-tune YOLO network, includes images/ labels/ data.yaml
templates/ -> images of all cards + tower/arena states to use for comparison/detection base later