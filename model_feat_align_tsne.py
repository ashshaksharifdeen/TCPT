#!/usr/bin/env python
# tsne_text_vs_ece.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# our ZeroshotCLIP helper and your MaPLe loader
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

    # we only need its .dm to know class names
    dummy = build_trainer(cfg_zs)
    dm    = dummy.dm

    zs = ZeroshotCLIP()
    zs.cfg    = cfg_zs
    zs.dm     = dm
    zs.device = device
    zs.build_model()  # populates zs.text_features

    frozen_feats = zs.text_features.cpu().numpy()  # [C, D]
    C, D = frozen_feats.shape
    # ----------------------------------------------------------------------------
    # 2) Your three fine‐tuned runs (all using the same template)
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
        # 1) run one batch to fill .textfeatures
        batch = next(iter(trainer.test_loader))
        imgs, labs = trainer.parse_batch_train(batch)
        with torch.no_grad():
            _ = trainer.model(imgs.to(device))
        tuned = trainer.model.textfeatures.cpu().numpy()
        all_feats[name] = tuned

        # 2) compute ECE over full test set
        preds, confs, labs = [], [], []
        with torch.no_grad():
            for batch in trainer.test_loader:
                imgs, lbls = trainer.parse_batch_train(batch)
                imgs = imgs.to(device)
                logits = trainer.model(imgs)
                probs  = torch.softmax(logits, dim=1).cpu()
                preds.append(probs.argmax(1))
                confs.append(probs.max(1).values)
                labs.append(lbls.cpu())
        preds = torch.cat(preds)
        confs = torch.cat(confs)
        labs  = torch.cat(labs)
        all_ece[name] = compute_ece(preds, confs, labs)
        print(f"    ECE = {all_ece[name]:.2f}%")

    # ----------------------------------------------------------------------------
    # 3) TSNE per‐method
    # ----------------------------------------------------------------------------
    perp = min(30, C - 1)
    tsne = TSNE(n_components=2, init="pca",  random_state=0, perplexity=perp)
    tsne_results = {m: tsne.fit_transform(all_feats[m]) for m in all_feats}

    # ----------------------------------------------------------------------------
    # 4) Plot them side‐by‐side
    # ----------------------------------------------------------------------------
    n = len(tsne_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, coords) in zip(axes[0], tsne_results.items()):
        x, y = coords[:,0], coords[:,1]
        ax.scatter(x, y, s=40, alpha=0.7)
        ece = all_ece.get(name, 0.0)
        ax.set_title(f"{name}\nECE = {ece:.1f}%", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Text‐Feature Geometry via t-SNE\n( Frozen vs. tuned )", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])

    out = "output/plots/tsne_text_vs_ece.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"\nSaved TSNE plot to {out}")


if __name__ == "__main__":
    main()
