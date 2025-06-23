#!/usr/bin/env python
# tsne_text_vs_ece.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from trainers.maple import ZeroshotCLIP
from train import extend_cfg

# register all the datasets you’ll need
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

# ——————————————————————————————————————————————
#  High-res + big fonts for NeurIPS
# ——————————————————————————————————————————————
plt.rcParams.update({
    'figure.dpi': 600,
    'font.size': 18,
    'axes.titlesize': 22,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

def compute_ece(preds, confs, labels, n_bins=20):
    """Expected Calibration Error (in %)."""
    bin_bounds = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(preds)
    for i in range(n_bins):
        lo, hi = bin_bounds[i], bin_bounds[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean()
            avg_conf = confs[mask].mean()
            ece += (avg_conf - acc).abs() * mask.sum().item() / N
    return ece * 100.0

def load_maple_trainer(exp_dir, prompt):
    """
    Build a MaPLe trainer, load its checkpoint, and return it.
    """
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.SEED = 1
    cfg.OUTPUT_DIR = exp_dir
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.CTX_INIT = prompt
    cfg.DATASET.NAME = "DescribableTextures"
    cfg.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg.freeze()

    trainer = build_trainer(cfg)
    trainer.load_model(exp_dir, epoch=5)
    trainer.model.eval()
    return trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------------
    # 1) Grab frozen CLIP text–features via your ZeroshotCLIP class
    # ----------------------------------------------------------------------------
    cfg_zs = get_cfg_default()
    extend_cfg(cfg_zs)
    cfg_zs.TRAINER.NAME = "ZeroshotCLIP"
    cfg_zs.DATASET.NAME = "DescribableTextures"
    cfg_zs.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg_zs.DATASET.SUBSAMPLE_CLASSES = "new"
    cfg_zs.DATASET.NUM_SHOTS = 16
    cfg_zs.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg_zs.freeze()

    dummy = build_trainer(cfg_zs)
    dm    = dummy.dm

    zs = ZeroshotCLIP()
    zs.cfg, zs.dm, zs.device = cfg_zs, dm, device
    zs.build_model()
    frozen_feats = zs.text_features.cpu().numpy()  # [C, D]
    C, _ = frozen_feats.shape

    # ----------------------------------------------------------------------------
    # 2) Three fine‐tuned runs (same template)
    # ----------------------------------------------------------------------------
    methods = {
        "Orthogonality":        "output/cosine/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed2",
        "L1 Alignment":         "output/l1align/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed2",
        "Text Moment-Matching": "output/textmomentum/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed2",
    }
    prompt = "a photo of a"

    all_feats = {"Frozen CLIP": frozen_feats}
    all_ece   = {"Frozen CLIP": 0.0}

    for name, exp_dir in methods.items():
        print(f"\n>>> Loading {name}")
        trainer = load_maple_trainer(exp_dir, prompt)
        trainer.model.to(device)

        # run one batch to fill .textfeatures
        batch = next(iter(trainer.test_loader))
        imgs, _ = trainer.parse_batch_train(batch)
        with torch.no_grad():
            _ = trainer.model(imgs.to(device))
        all_feats[name] = trainer.model.textfeatures.cpu().numpy()

        # compute ECE over full test set
        preds, confs, labs = [], [], []
        with torch.no_grad():
            for batch in trainer.test_loader:
                imgs, lbls = trainer.parse_batch_train(batch)
                probs  = torch.softmax(trainer.model(imgs.to(device)), dim=1).cpu()
                preds.append(probs.argmax(1))
                confs.append(probs.max(1).values)
                labs.append(lbls.cpu())
        preds = torch.cat(preds)
        confs = torch.cat(confs)
        labs  = torch.cat(labs)
        all_ece[name] = compute_ece(preds, confs, labs)
        print(f"    ECE = {all_ece[name]:.2f}%")

    methods = list(all_feats.keys())

    # ----------------------------------------------------------------------------
    # 3) Stack features, run t-SNE
    # ----------------------------------------------------------------------------
    X = np.vstack([all_feats[m] for m in methods])  # [M*C, D]
    perp = min(30, (len(methods)*C)//3)
    X2 = TSNE(n_components=2, init="pca", random_state=0, perplexity=perp).fit_transform(X)

    coords = {}
    idx = 0
    for m in methods:
        coords[m] = X2[idx:idx+C]
        idx += C

    # ----------------------------------------------------------------------------
    # 4) Plot scatter — legend inside upper‐right, larger figure
    # ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16,16))  # <-- increased size

    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        xy = coords[m]
        label = m if m == "Frozen CLIP" else f"{m} (ECE={all_ece[m]:.1f}%)"
        ax.scatter(
            xy[:,0], xy[:,1],
            s=400,               # slightly larger markers
            alpha=0.8,
            color=cmap(i),
            label=label
        )

    ax.set_title("Combined t-SNE of Text Features",fontsize=36)
    ax.set_xticks([]); ax.set_yticks([])

    # Expand axes to cover almost the entire figure
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)

    # Legend inside axes, upper-right, with white background
    ax.legend(
        loc='upper right',
        fontsize=24,
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='black'
    )

    # Save high‐res figure
    out = "output/plots/tsne_text_vs_ece_combined.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f"Saved combined t-SNE plot to {out}")

if __name__ == "__main__":
    main()
