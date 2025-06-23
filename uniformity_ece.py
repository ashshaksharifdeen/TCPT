#!/usr/bin/env python
import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# your extension from train.py
from train import extend_cfg  

# Import datasets so that they are registered.
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

# -------------------------------------------------------------------
# ECE & Uniformity helpers
# -------------------------------------------------------------------
def compute_ece(preds, confs, labels, n_bins=20):
    """Compute Expected Calibration Error."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(preds)
    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b+1]
        mask = (confs > low) & (confs <= high)
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean()
            avg_conf = confs[mask].mean()
            ece += (avg_conf - acc).abs() * mask.sum().item() / total
    return ece

def compute_uniformity(features, subsample=2000):
    """
    Uniformity = mean_{i!=j} exp(-2 ||z_i - z_j||^2)
    We sub-sample if the full set is too large.
    """
    N, D = features.shape
    if N > subsample:
        idx = random.sample(range(N), subsample)
        feats = features[idx].float()   # cast to float32
        M = subsample
    else:
        feats = features.float()        # cast to float32
        M = N

    diffs = feats.unsqueeze(1) - feats.unsqueeze(0)  # (M,M,D)
    sqd = (diffs**2).sum(-1)                         # (M,M)
    mask = torch.eye(M, device=sqd.device).bool()
    sqd = sqd.masked_fill(mask, float('inf'))

    u = torch.exp(-2 * sqd).sum() / (M * (M - 1))
    return u

# -------------------------------------------------------------------
# Load a trained MaPLe checkpoint into a fresh trainer
# -------------------------------------------------------------------
def load_trainer_from_dir(exp_dir, dataset_name, prompt):
    """
    exp_dir: top-level exp dir that contains `MultiModalPromptLearner/model.pth.tar-5`
    dataset_name: e.g. "OxfordFlowers"
    prompt: string to set as cfg.TRAINER.MAPLE.CTX_INIT
    """
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.SEED = 1
    cfg.OUTPUT_DIR = exp_dir
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.CTX_INIT = prompt
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"

    trainer = build_trainer(cfg)
    trainer.load_model(exp_dir, epoch=5)
    trainer.model.eval()
    return trainer

# -------------------------------------------------------------------
# Main comparison
# -------------------------------------------------------------------
def main():
    """
    Assumes you have 3 experiment directories,
    each containing `.../MultiModalPromptLearner/model.pth.tar-5`.
    """
    # paths to your three trained models; all use the same prompt template:
    methods = {
        "Orthogonality": (
            "output/cosine/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed2",
            "a photo of a lorry"
        ),
        "L1 Alignment": (
            "output/l1align/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed2",
            "a photo of a"
        ),
        "Text Moment-Matching": (
            "output/textmomentum/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed2",
            "a photo of a"
        ),
    }
    dataset_name = "DescribableTextures"

    results = {}
    for method_name, (exp_dir, prompt) in methods.items():
        print(f"\n=== Evaluating {method_name} ===")
        trainer = load_trainer_from_dir(exp_dir, dataset_name, prompt)

        all_preds, all_confs, all_labels, all_feats = [], [], [], []

        with torch.no_grad():
            for batch in trainer.test_loader:
                imgs, labs = trainer.parse_batch_train(batch)
                logits = trainer.model(imgs)            # sets .imfeatures
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)
                confs  = probs.max(dim=1).values

                all_preds .append(preds.cpu())
                all_confs .append(confs.cpu())
                all_labels.append(labs.cpu())
                all_feats .append(trainer.model.textfeatures.cpu())

        all_preds  = torch.cat(all_preds,  dim=0)
        all_confs  = torch.cat(all_confs,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_feats  = torch.cat(all_feats,  dim=0)

        ece = compute_ece(all_preds, all_confs, all_labels, n_bins=20) * 100.0
        uni = compute_uniformity(all_feats, subsample=2000)

        print(f"{method_name:<20} →   ECE = {ece:.2f}%,  Uniformity = {uni:.6f}")
        results[method_name] = (ece, uni)

    # --- plot them all on a single figure ---
    plt.figure(figsize=(8, 6))
    markers = {"Orthogonality": "o", "L1 Alignment": "s", "Text Moment-Matching": "^"}
    colors  = {"Orthogonality": "C0", "L1 Alignment": "C1", "Text Moment-Matching": "C2"}

    for name, (ece, uni) in results.items():
        plt.scatter(uni, ece,
                    marker=markers[name],
                    color=colors[name],
                    s=200,
                    label=name,
                    edgecolor='k',
                    linewidth=1.0)

    plt.xlabel("Uniformity", fontsize=12)
    plt.ylabel("Expected Calibration Error (ECE %)", fontsize=12)
    plt.title("Uniformity vs ECE Across Methods", pad=12, fontsize=14)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = "output/plots/compare_uniformity_ece.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved figure to {out_path}")

if __name__ == "__main__":
    main()
