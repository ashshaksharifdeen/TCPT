#!/usr/bin/env python
import os
import torch
import random
import glob
import numpy as np
import matplotlib.pyplot as plt

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg   # your config‐extension fn

# -----------------------------------------------------------------------------
# ECE helper
# -----------------------------------------------------------------------------
def compute_ece(preds, confs, labels, n_bins=20):
    bin_bounds = torch.linspace(0, 1, n_bins+1)
    ece = 0.0
    total = len(preds)
    for b in range(n_bins):
        low, high = bin_bounds[b], bin_bounds[b+1]
        mask = (confs > low) & (confs <= high)
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean()
            avg_conf = confs[mask].mean()
            ece += (avg_conf - acc).abs() * mask.sum().item() / total
    return ece.item()

# -----------------------------------------------------------------------------
# Modality‐Gap helper
# -----------------------------------------------------------------------------
def compute_modality_gap(trainer):
    """
    Gap = || mean_image_feats  -  mean_text_feats ||_2^2
    where text_feats are the tuned‐prompt embeddings for each sample's label.
    """
    trainer.model.eval()
    all_img, all_txt = [], []
    with torch.no_grad():
        for batch in trainer.test_loader:
            imgs, labs = trainer.parse_batch_train(batch)
            _ = trainer.model(imgs)  # populates .imfeatures and .textfeatures
            z_img = trainer.model.imfeatures       # [B, D]
            z_cls = trainer.model.textfeatures      # [C, D]
            z_txt = z_cls[labs]                    # pick per‐sample text embedding
            all_img.append(z_img.cpu())
            all_txt.append(z_txt.cpu())

    all_img = torch.cat(all_img, dim=0)
    all_txt = torch.cat(all_txt, dim=0)
    mu_img = all_img.mean(dim=0)
    mu_txt = all_txt.mean(dim=0)
    gap = (mu_img - mu_txt).pow(2).sum().item()
    return gap

# -----------------------------------------------------------------------------
# Load MaPLe trainer from an experiment directory
# -----------------------------------------------------------------------------
def load_trainer(exp_dir, prompt, dataset_name="OxfordFlowers"):
    """
    exp_dir: path to the folder containing 
             MultiModalPromptLearner/model.pth.tar-5
    prompt: the CTX_INIT string you used for that experiment
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

    trainer = build_trainer(cfg)
    trainer.load_model(exp_dir, epoch=5)
    trainer.model.eval()
    return trainer

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # === replace with your actual directories & prompts ===
    methods = {
      "Orthogonality"       : ("output/.../orthogonality_exp/seed1",       "a prompt for orthogonality"),
      "L1 Alignment"        : ("output/.../l1_alignment_exp/seed1",        "a prompt for L1 alignment"),
      "Text Moment‐Matching": ("output/.../moment_matching_exp/seed1",     "a prompt for moment‐matching"),
    }
    dataset_name = "OxfordFlowers"

    results = {}
    for name, (exp_dir, prompt) in methods.items():
        print(f"→ Evaluating {name}")
        trainer = load_trainer(exp_dir, prompt, dataset_name)

        # collect preds, confs, labels
        all_preds, all_confs, all_labels = [], [], []
        with torch.no_grad():
            for batch in trainer.test_loader:
                imgs, labs = trainer.parse_batch_train(batch)
                logits = trainer.model(imgs)
                probs  = torch.softmax(logits, dim=1)
                all_preds .append(probs.argmax(dim=1).cpu())
                all_confs .append(probs.max(dim=1).values.cpu())
                all_labels.append(labs.cpu())
        preds  = torch.cat(all_preds,  dim=0)
        confs  = torch.cat(all_confs,  dim=0)
        labels = torch.cat(all_labels, dim=0)

        # compute metrics
        ece = compute_ece(preds, confs, labels, n_bins=20)
        gap = compute_modality_gap(trainer)

        print(f"   ECE = {ece:.3f},  Modality Gap = {gap:.6f}")
        results[name] = (ece, gap)

    # === plot Modality Gap vs ECE ===
    plt.figure(figsize=(8,6))
    markers = {"Orthogonality":"o","L1 Alignment":"s","Text Moment‐Matching":"^"}
    colors  = {"Orthogonality":"C0","L1 Alignment":"C1","Text Moment‐Matching":"C2"}

    for name,(ece,gap) in results.items():
        plt.scatter(gap, ece,
                    marker=markers[name],
                    color=colors[name],
                    s=200,
                    label=name,
                    edgecolor='k', linewidth=1.2)

    plt.xlabel("Modality Gap\n$\|\mu_{img}-\mu_{txt}\|_2^2$", fontsize=12)
    plt.ylabel("Expected Calibration Error (ECE %)", fontsize=12)
    plt.title("Modality Gap vs ECE Across Methods", fontsize=14, pad=14)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.tight_layout()

    out = "output/plots/modality_gap_vs_ece.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"\n→ Saved plot to {out}")

if __name__=="__main__":
    main()
