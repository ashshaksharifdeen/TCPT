#!/usr/bin/env python
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg   # your cfg extension

# register all your datasets (so trainer.test_loader works)
import datasets.oxford_flowers
import datasets.dtd
# … etc

def load_trainer_from_dir(exp_dir, dataset_name, prompt):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.SEED = 1
    cfg.OUTPUT_DIR = exp_dir
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.CTX_INIT = prompt
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"

    trainer = build_trainer(cfg)
    trainer.load_model(exp_dir, epoch=5)
    trainer.model.eval()
    return trainer

def per_class_uniformity(text_feats):
    """
    text_feats: torch.Tensor[C, D] (float32)
    returns: numpy array of length C,
      u_i = (1/(C-1)) sum_{j!=i} exp(-2*||z_i - z_j||^2)
    """
    with torch.no_grad():
        C, D = text_feats.shape
        diffs = text_feats.unsqueeze(1) - text_feats.unsqueeze(0)  # (C,C,D)
        sqd   = (diffs**2).sum(-1)                                  # (C,C)
        mask  = torch.eye(C, device=sqd.device).bool()              # mask diag
        sqd[mask] = float('inf')
        E = torch.exp(-2 * sqd)    # (C,C)
        u = E.sum(dim=1) / (C - 1)  # (C,)
    return u.cpu().numpy()

def main():
    methods = {
            "Orthogonality":             "output/cosine/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed1",        
            "L1 Alignment":              "output/l1align/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed1",   
            "Text Moment-Matching": 
            "output/textmomentum/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed1",
 
    }
    dataset_name = "DescribableTextures"
    prompt       = "a photo of a"

    violin_data = []
    labels      = []

    for name, exp_dir in methods.items():
        print(f"\nLoading {name} from {exp_dir}")
        trainer = load_trainer_from_dir(exp_dir, dataset_name, prompt)

        # Run exactly ONE forward pass to populate .textfeatures
        for batch in trainer.test_loader:
            imgs, _ = trainer.parse_batch_train(batch)
            _ = trainer.model(imgs)            # this sets .textfeatures internally
            tf = trainer.model.textfeatures    # [C, D]
            break

        uniformities = per_class_uniformity(tf.detach().float())
        violin_data.append(uniformities)
        labels.append(name)

    # --- now plot the violins ---
    plt.figure(figsize=(9,6))
    parts = plt.violinplot(violin_data,
                           showmeans=False,
                           showmedians=True,
                           showextrema=False)

    colors = ["C0","C1","C2"]
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
        pc.set_alpha(0.7)

    parts['cmedians'].set_color('k')
    parts['cmedians'].set_linewidth(2)

    plt.xticks(np.arange(1, len(labels)+1), labels, fontsize=12)
    plt.ylabel("Per-Class Uniformity $u_i$", fontsize=13)
    plt.title("Distribution of Text-Feature Uniformities\nAcross Methods", fontsize=16, pad=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = "output/plots/text_uniformity_violin.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300)
    print("Saved figure to", out)

if __name__ == "__main__":
    main()
