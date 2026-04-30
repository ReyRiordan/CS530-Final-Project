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

### State Representations

Environment: construct state tensor (C, H, W) where H=32 and W=18 (arena tiling dimensions) and C=16+ feature channels
- Channel 0-7: ally troops (1 if that troop at that tile, 0 if not)
- Channel 8-15: enemy troops
- Elixir + tower HP: each normalized to [0,1]
- Cards in hand: one-hot encoded vector x 4

Actions: discrete action space of whether to play card, which card to play, where to play card
- Discrete choice of 5 actions: card 1, card 2, card 3, card 4, "wait"
- Discrete choice of 16x18 = 288 tiles (deck contains only troops -> can only play in ally half of arena) *Actually only 224 valid tiles here due to towers in the way and such
- NOTE: more valid tiles open up to place troops after taking an enemy tower, but we are ignoring this for simplicity

### Agent Policy

Bootstrapping on human behavior (imitation learning):
- Python script that records data while human plays on emulator: record environment state every second, record actions taken (attach actions to nearest state for dataset)
- Human plays N matches against arena 1 training camp bot (try to play predictably, use similar counters and build up pushes in similar ways)
- Train CNN to predict human actions based on environment state
- Environment state input: run 16 channels through conv layers, then flatten + concatenate non-channel input before dense layers
- Action state output: use two heads (card head, tile head) -> card head chooses one of 4 cards in hand or "wait", then that choice (if not "wait") is fed as one-hot concat on feature vector into tile head to choose out of 224 valid tiles
- Two cross-entropy losses (card, tile) -> zero out tile loss when card not chosen, normalize, then add for total loss


Reinforcement Learning:
- Large environment and action state spaces -> use Proximal Policy Optimization (PPO)
- Reward function: calculate change in tower health (positive reward for damage dealt, negative for damage taken)
- Also reward/penalize other actions such as leaking elixir (penalty), trying to place card when not enough elixir for it (penalty), destroying enemy tower (when hits 0 HP, huge reward), letting ally tower get destroyed (huge penalty), win game (huge reward), lose game (huge penalty) -> require more harness / computer vision components?
- Create additional harness components that automatically detect game end and start new training game
- Use human action bootstrapped CNN as policy, run PPO on these rewards for however much time is available (maybe run overnight?)
- **ISSUE**: PPO is actor-critic learning, CNN is actor but no critic trained during imitation learning -> critic will start out ass and might ruin the actor too -> split CNN into shared feature extractor? freeze actor to let critic catch-up for a bit? feed human dataset directly into PPO to do offline learning first (SB3 imitation)? Generative Adversarial Imitation Learning (GAIL)? *PROB DOING CRITIC WARMUP CUZ EASIEST
