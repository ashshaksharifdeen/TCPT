#!/usr/bin/env python
import os
import os.path as osp
import cv2
import torch
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

# DASSL & custom trainer stuff
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg
import trainers.maple  # ensure your MaPLe trainer is registered

def save_attention_hook(module, input_, output_, attention_cache):
    """
    Hook function to grab the self-attention weights.
    Expects the forward call to return (attn_output, attn_weights).
    """
    if isinstance(output_, tuple) and len(output_) == 2:
        attn = output_[1]
        attention_cache["attn"] = attn.detach().cpu()
    else:
        raise ValueError("Cannot find attention in output. "
                         "Adjust indexing/hook or ensure need_weights=True in the attention call.")

def main(args):
    # Build config and load trainer.
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.SEED = 1
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample_classes
    cfg.freeze()

    trainer = build_trainer(cfg)
    trainer.load_model(osp.join(args.model_dir, "seed1"), epoch=args.load_epoch)
    trainer.model.eval()

    # Optionally, set test_loader.num_workers = 0 to avoid multiprocessing issues.
    if hasattr(trainer.test_loader, "num_workers"):
        trainer.test_loader.num_workers = 0

    # Create the output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Monkey patch the attention module to force need_weights=True ---
    visual_transformer = trainer.model.image_encoder.transformer.resblocks[-1]
    old_forward = visual_transformer.attn.forward
    def new_forward(self, query, key, value, **kwargs):
        kwargs["need_weights"] = True
        return old_forward(query, key, value, **kwargs)
    visual_transformer.attn.forward = new_forward.__get__(visual_transformer.attn, type(visual_transformer.attn))
    # --------------------------------------------------------------------

    # Register the hook once; we'll clear our cache before processing each sample.
    attention_cache = {}
    handle = visual_transformer.attn.register_forward_hook(
        lambda mod, inp, out: save_attention_hook(mod, inp, out, attention_cache)
    )

    # Collect 10 samples (each sample is a tuple of (input_tensor, ground_truth_label))
    samples = []
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)
            bs = images.shape[0]
            for i in range(bs):
                samples.append((images[i:i+1], labels[i:i+1]))
                if len(samples) >= 10:
                    break
            if len(samples) >= 10:
                break

    print(f"Collected {len(samples)} samples for visualization.")

    # Process each sample.
    for idx, (input_tensor, gt) in enumerate(samples):
        # Ensure input is float and disable gradients.
        input_tensor = input_tensor.float()
        input_tensor.requires_grad_(False)

        # Convert image tensor to numpy (in RGB) for visualization.
        img_np = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        h, w, _ = img_np.shape
        print(f"Sample {idx}: Original image shape (h,w):", h, w)

        # Clear the attention cache for this sample.
        attention_cache.clear()
        # Forward pass to trigger the hook and capture model output.
        with torch.no_grad():
            output = trainer.model(input_tensor)
        # Compute predicted label and softmax confidence.
        pred_label = output.argmax(dim=1).item()
        gt_label = gt.item()
        softmax_output = torch.softmax(output, dim=1)
        confidence = softmax_output[0, pred_label].item()
        if pred_label == gt_label:
            line1 = f"Correct: GT {gt_label}, Pred {pred_label}"
        else:
            line1 = f"Incorrect: GT {gt_label}, Pred {pred_label}"
        line2 = f"Conf: {confidence:.2f}"
        print(f"Sample {idx}: {line1} | {line2}")

        if "attn" not in attention_cache:
            raise ValueError(f"Attention weights were not retrieved for sample {idx}. "
                             "Ensure that need_weights=True is set or adjust your hooking approach.")
        
        # Retrieve attention weights; expected shape: [batch, n_heads, seq_len, seq_len]
        attn_weights = attention_cache["attn"].squeeze(0)  # remove batch dimension
        print(f"Sample {idx}: Retrieved attention weights with shape:", attn_weights.shape)
        if attn_weights.ndim == 3:
            attn_weights = attn_weights.mean(dim=0)  # now shape: [seq_len, seq_len]
        elif attn_weights.ndim == 2:
            pass
        elif attn_weights.ndim == 1:
            pass
        else:
            raise ValueError("Unexpected attention tensor dimensions: {}".format(attn_weights.ndim))

        # Determine sequence length and patch count.
        seq_len = attn_weights.shape[0]
        K = trainer.model.prompt_learner.n_ctx  # number of extra tokens (e.g., prompt tokens)
        patch_count = seq_len - 1 - K
        if patch_count <= 0:
            raise ValueError(f"Not enough tokens remain for patches for sample {idx}. Got seq_len={seq_len} with K={K}")
        print(f"Sample {idx}: seq_len: {seq_len}, K: {K}, patch_count: {patch_count}")

        # Extract CLS token's attention over patch tokens.
        if attn_weights.ndim == 2:
            patch_attn = attn_weights[0, 1 : 1 + patch_count]
        elif attn_weights.ndim == 1:
            patch_attn = attn_weights[1 : 1 + patch_count]
        else:
            raise ValueError("Unexpected dimensions when extracting patch attention.")
        if isinstance(patch_attn, torch.Tensor):
            patch_attn = patch_attn.detach().numpy()
        print(f"Sample {idx}: patch_attn shape:", patch_attn.shape)

        # Assume patch tokens form a square grid (e.g., 14x14)
        patch_side = int(math.sqrt(patch_count))
        if patch_side * patch_side != patch_count:
            raise ValueError(f"Sample {idx}: patch_count {patch_count} is not a perfect square. Adjust for your input resolution or patching strategy.")
        try:
            patch_attn_2d = patch_attn.reshape(patch_side, patch_side)
        except Exception as e:
            print(f"Sample {idx}: Error reshaping patch_attn with shape", patch_attn.shape, "to", (patch_side, patch_side))
            raise e

        # Convert to contiguous float32 array.
        patch_attn_2d = np.ascontiguousarray(patch_attn_2d.astype(np.float32))
        print(f"Sample {idx}: patch_attn_2d shape:", patch_attn_2d.shape)
        print(f"Sample {idx}: patch_attn_2d dtype:", patch_attn_2d.dtype)

        # Normalize the attention map.
        patch_attn_2d = patch_attn_2d / (patch_attn_2d.max() + 1e-8)
        # Try resizing using INTER_CUBIC; if that fails, fallback to INTER_LINEAR.
        try:
            attn_map_up = cv2.resize(patch_attn_2d, (w, h), interpolation=cv2.INTER_CUBIC)
        except cv2.error as e:
            print(f"Sample {idx}: cv2.resize with INTER_CUBIC failed:", e)
            try:
                attn_map_up = cv2.resize(patch_attn_2d, (w, h), interpolation=cv2.INTER_LINEAR)
                print(f"Sample {idx}: Falling back to INTER_LINEAR.")
            except cv2.error as e2:
                print(f"Sample {idx}: cv2.resize with INTER_LINEAR also failed:", e2)
                raise e2

        heatmap = cv2.applyColorMap((attn_map_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # Convert heatmap from BGR to RGB.
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * img_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        # Convert overlay to BGR (as expected by OpenCV) for annotation.
        annotated = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Define annotation positions and font parameters.
        text_color = (0, 255, 0) if pred_label == gt_label else (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line1_pos = (10, 30)
        line2_pos = (10, 60)
        cv2.putText(annotated, line1, line1_pos, font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(annotated, line2, line2_pos, font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Save the results for this sample.
        orig_fname = osp.join(args.output_dir, f"original_image_{idx}.png")
        attn_fname = osp.join(args.output_dir, f"attention_map_{idx}.png")
        cv2.imwrite(orig_fname, cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(attn_fname, annotated)
        print(f"[✓] Saved sample {idx} original image as {orig_fname}")
        print(f"[✓] Saved sample {idx} attention map as {attn_fname}")

    # Remove the hook after processing all samples.
    handle.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--config-file", type=str, required=True, help="Trainer config YAML")
    parser.add_argument("--dataset-config-file", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--model-dir", type=str, required=True, help="Base dir with seed subfolders")
    parser.add_argument("--load-epoch", type=int, default=5, help="Epoch to load")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--subsample-classes", type=str, default="new", help="Which class split to use")
    args = parser.parse_args()
    main(args)
