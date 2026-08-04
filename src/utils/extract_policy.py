import sys
import torch

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_policy.py <checkpoint_path> <output_path>")
        sys.exit(1)

    checkpoint_path, output_path = sys.argv[1], sys.argv[2]
    state = torch.load(checkpoint_path, map_location="mps")
    torch.save(state["online_net"], output_path)
    print(f"Saved weights to {output_path}")
