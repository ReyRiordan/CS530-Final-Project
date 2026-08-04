import random
import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), # -> 128
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 39, 256), # 39 is flat_vector
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 33), # Q estimates for 33 actions
        )

    def forward(self, troop_tensor, flat_vector):
        # troop_tensor: (B, 16, 32, 18), flat_vector: (B, 39)
        spatial = self.conv(troop_tensor) # (B, 128)
        combined = torch.cat([spatial, flat_vector], dim=1) # (B, 167)

        return self.head(combined) # (B, 33)


def load_policy(checkpoint_path, device="mps"):
    model = PolicyNetwork()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model


def select_action(model, troop_tensor, flat_vector, epsilon=0.0, device="mps"):
    if random.random() < epsilon:
        return random.randrange(33)
    
    # Add batch dim of 1
    troop_tensor_input = torch.from_numpy(troop_tensor).unsqueeze(0).to(device) # (1, 16, 32, 18)
    flat_vector_input = torch.from_numpy(flat_vector).unsqueeze(0).to(device) # (1, 39)

    with torch.no_grad():
        q_estimates = model(troop_tensor_input, flat_vector_input)

    return int(q_estimates.argmax(dim=1).item())
