#!/usr/bin/env python
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from dassl.config import get_cfg_default
from dassl.engine   import build_trainer
from trainers.maple import ZeroshotCLIP
from train          import extend_cfg

# register any datasets you need:
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
    """Expected Calibration Error (%)"""
    # keep everything on preds.device
    bin_bounds = torch.linspace(0, 1, n_bins + 1, device=preds.device)
    ece = 0.0
    N = len(preds)
    for i in range(n_bins):
        lo, hi = bin_bounds[i], bin_bounds[i+1]
        mask = (confs > lo) & (confs <= hi)
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean()
            avg_conf = confs[mask].mean()
            ece += (avg_conf - acc).abs() * mask.sum().item() / N
    return ece * 100.0

def load_maple_trainer(exp_dir, prompt, device):
    """Build & load a MaPLe trainer from exp_dir/… onto `device`."""
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.SEED = 1
    cfg.OUTPUT_DIR = exp_dir
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.CTX_INIT = prompt
    cfg.DATASET.NAME = "OxfordFlowers"
    cfg.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg.freeze()

    trainer = build_trainer(cfg)
    trainer.load_model(exp_dir, epoch=5)
    trainer.model.to(device)
    trainer.model.eval()
    return trainer

def main():
    # Force CUDA device 2
    torch.cuda.set_device(2)
    device = torch.device("cuda:2")

    # 1) Obtain frozen CLIP text‐features
    cfg_zs = get_cfg_default()
    extend_cfg(cfg_zs)
    cfg_zs.TRAINER.NAME = "ZeroshotCLIP"
    cfg_zs.DATASET.NAME = "OxfordFlowers"
    cfg_zs.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg_zs.DATASET.SUBSAMPLE_CLASSES = "new"
    cfg_zs.DATASET.NUM_SHOTS = 16
    cfg_zs.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg_zs.freeze()

    dummy = build_trainer(cfg_zs)
    zs_tr  = ZeroshotCLIP()
    zs_tr.cfg    = cfg_zs
    zs_tr.dm     = dummy.dm
    zs_tr.device = device
    zs_tr.build_model()
    zs_tr.clip_model.to(device)

    frozen_feats = zs_tr.text_features.to(device).cpu().numpy()

    # 2) Discover all prompt variants
    ROOT_DIR = "output/base2new/train_base/oxford_flowers/shots_16/MaPLe"
    pattern  = os.path.join(ROOT_DIR, "vit_b16_c2_ep5_batch4_2ctx_*", "seed1")
    exp_dirs = sorted(glob.glob(pattern))

    distances = []
    eces      = []
    names     = []

    for exp_dir in exp_dirs:
        prompt_token = os.path.basename(os.path.dirname(exp_dir))
        prompt = prompt_token.replace("vit_b16_c2_ep5_batch4_2ctx_", "").replace("_", " ")
        name   = prompt

        print(f"\n→ Processing prompt = “{prompt}”")

        trainer = load_maple_trainer(exp_dir, prompt, device)

        # one forward to fill textfeatures
        batch = next(iter(trainer.test_loader))
        imgs, _ = trainer.parse_batch_train(batch)
        imgs = imgs.to(device)
        with torch.no_grad():
            _ = trainer.model(imgs)
        tuned_feats = trainer.model.textfeatures.to(device).cpu().numpy()

        # mean ℓ₂ distance
        dist = float(np.abs(tuned_feats - frozen_feats).sum(axis=1).mean())
        distances.append(dist)

        # compute ECE (convert to float.item())
        all_preds, all_confs, all_labs = [], [], []
        with torch.no_grad():
            for b in trainer.test_loader:
                imgs, labs = trainer.parse_batch_train(b)
                imgs = imgs.to(device)
                logits = trainer.model(imgs)
                probs  = torch.softmax(logits, dim=1).cpu()
                all_preds.append(probs.argmax(1))
                all_confs.append(probs.max(1).values)
                all_labs.append(labs)
        preds = torch.cat(all_preds).to(device)
        confs = torch.cat(all_confs).to(device)
        labs  = torch.cat(all_labs).to(device)
        ece_tensor = compute_ece(preds, confs, labs)
        ece = ece_tensor.item()
        eces.append(ece)

        names.append(name)
        print(f"   → distance = {dist:.3f},  ECE = {ece:.2f}%")

    # 3) Plot distance vs ECE
    plt.figure(figsize=(7,6))
    for x,y,lbl in zip(distances, eces, names):
        plt.scatter(x, y, s=80)
        plt.text(x+1e-3, y+0.2, lbl, fontsize=8)

    # linear fit
    m,b = np.polyfit(distances, eces, 1)
    xs  = np.linspace(min(distances), max(distances), 100)
    plt.plot(xs, m*xs+b, "--", color="gray", label=f"fit: y={m:.2f}x+{b:.2f}")

    plt.xlabel("Mean ℓ₂ distance to frozen CLIP", fontsize=12)
    plt.ylabel("ECE (%)", fontsize=12)
    plt.title("Prompt‐tuning: Embedding‐Drift vs Calibration Error", fontsize=14)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("output/plots", exist_ok=True)
    out = "output/plots/dist_vs_ece_allprompts.png"
    plt.savefig(out, dpi=300)
    print(f"\nSaved plot to {out}")

if __name__ == "__main__":
    main()
