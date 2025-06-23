#!/usr/bin/env python
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg   # your patch from train.py

# register all your datasets so trainer.test_loader works
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

# ----------------------------------------------------------------------------
# ECE & Uniformity helpers (same as before)
# ----------------------------------------------------------------------------
def compute_ece(preds, confs, labels, n_bins=20):
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
    return ece * 100.0  # return as percentage

def per_class_uniformity(text_feats):
    """
    text_feats: [C, D] float32
    returns: numpy array of length C, each
      u_i = (1/(C-1)) sum_{j!=i} exp(-2 * ||z_i - z_j||^2)
    """
    with torch.no_grad():
        C, D = text_feats.shape
        diffs = text_feats.unsqueeze(1) - text_feats.unsqueeze(0)  # (C,C,D)
        sqd   = (diffs**2).sum(-1)                                  # (C,C)
        mask  = torch.eye(C, device=sqd.device).bool()
        sqd[mask] = float('inf')
        E = torch.exp(-2 * sqd)    # (C,C)
        u = E.sum(dim=1) / (C - 1)  # (C,)
    return u.cpu().numpy()

# ----------------------------------------------------------------------------
# load a MaPLe checkpoint into a fresh trainer
# ----------------------------------------------------------------------------
def load_trainer(exp_dir, dataset_name, prompt):
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

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    methods = {
        "Orthogonality":      "output/cosine/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed3",
        "L1 Alignment":       "output/l1align/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed3",
        "Text Moment-Matching":"output/textmomentum/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed3",
    }
    dataset_name = "DescribableTextures"
    prompt       = "a photo of a {}"

    all_uni = []
    ece_dict = {}

    for name, exp_dir in methods.items():
        print(f"→ Loading {name}")
        trainer = load_trainer(exp_dir, dataset_name, prompt)
        # --- 1) get ECE on the test split ---
        preds, confs, labs = [], [], []
        with torch.no_grad():
            for batch in trainer.test_loader:
                imgs, labels = trainer.parse_batch_train(batch)
                logits = trainer.model(imgs)      # populates .textfeatures inside
                probs  = torch.softmax(logits, dim=1)
                preds .append(probs.argmax(dim=1).cpu())
                confs .append(probs.max(dim=1).values.cpu())
                labs  .append(labels.cpu())
        preds = torch.cat(preds)
        confs = torch.cat(confs)
        labs  = torch.cat(labs)
        ece = compute_ece(preds, confs, labs)
        ece_dict[name] = ece

        # --- 2) get text‐prompt features and per‐class uniformities ---
        # inside MaPLe, after a forward on *any* batch, trainer.model.textfeatures exists,
        # but we can simply pull the *zero‐shot* textfeatures directly:
        tf = trainer.model.textfeatures.detach().float()  # shape [C, D]
        uni = per_class_uniformity(tf)
        all_uni.append(uni)

    # ----------------------------------------------------------------------------
    # Plot all three KDEs side by side, labeling each with its ECE
    # ----------------------------------------------------------------------------
    plt.figure(figsize=(10,6))
    for uni, name in zip(all_uni, methods.keys()):
        sns.kdeplot(
            uni,
            fill=True,
            alpha=0.5,
            lw=2,
            label=f"{name} (ECE={ece_dict[name]:.2f}%)"
        )

    plt.xlabel("Per‐Class Uniformity $u_i$",   fontsize=14)
    plt.ylabel("Density",                     fontsize=14)
    plt.title ("PDFs of Text‐Feature Uniformities\nacross Prompt‐Tuning Methods", fontsize=16, pad=14)
    plt.legend(loc="upper left", frameon=True, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("output/plots", exist_ok=True)
    out_path = "output/plots/text_uniformity_pdfs.png"
    plt.savefig(out_path, dpi=300)
    print("Saved figure to", out_path)

if __name__ == "__main__":
    main()
