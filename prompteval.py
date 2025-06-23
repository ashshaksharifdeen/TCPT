#!/usr/bin/env python
import argparse
import os
import torch

def extract_prompt(model_dir, prompt_name):
    # Construct the checkpoint path.
    # Here we assume the prompt learner’s model is saved under "MultiModalPromptLearner/model-best.pth.tar".
    checkpoint_path = os.path.join(model_dir, "MultiModalPromptLearner", "model-best.pth.tar-5")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if prompt_name not in state_dict:
        print(f"Parameter '{prompt_name}' not found in checkpoint.")
        return
    prompt_value = state_dict[prompt_name]
    print(f"Learned prompt values for '{prompt_name}':")
    print(prompt_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract trained prompt parameter from saved model checkpoint")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory where the model is saved")
    parser.add_argument("--prompt-name", type=str, required=True, help="Name of the prompt parameter (e.g., 'prompt_learner.ctx')")
    args = parser.parse_args()
    extract_prompt(args.model_dir, args.prompt_name)
