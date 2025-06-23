#!/usr/bin/env python3
# zero_shot_exp.py

import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg
import trainers.zsclip  # ensure ZeroshotCLIP is registered

def ECE_Loss(num_bins, preds, confs, correct):
    bounds = torch.linspace(0, 1, num_bins + 1)
    lowers, uppers = bounds[:-1], bounds[1:]
    bin_acc = [0.0]*num_bins
    bin_conf = [0.0]*num_bins
    bin_cnt = [0]*num_bins

    for p, c, corr in zip(preds, confs, correct):
        for b, (lo, hi) in enumerate(zip(lowers, uppers)):
            if lo < c <= hi:
                bin_cnt[b]  += 1
                bin_acc[b]  += corr
                bin_conf[b] += c

    for b in range(num_bins):
        if bin_cnt[b] > 0:
            bin_acc[b]  /= bin_cnt[b]
            bin_conf[b] /= bin_cnt[b]

    ece = sum(
        abs(bin_acc[b] - bin_conf[b]) * (bin_cnt[b] / len(preds))
        for b in range(num_bins)
    )
    return ece, bin_acc

def compute_metrics(preds, labels, confs):
    correct = (preds == labels).tolist()
    ece_val, bin_acc = ECE_Loss(20, preds.tolist(), confs.tolist(), correct)
    acc = sum(correct) / len(correct)
    return acc * 100, ece_val * 100, bin_acc

def run_one(cfg, output_dir, split_name, root):
    # 1) override root and splits
    cfg.defrost()
    cfg.DATASET.ROOT = root
    cfg.DATASET.SUBSAMPLE_CLASSES = split_name
    cfg.TRAINER.NAME = "ZeroshotCLIP"
    # disable multiprocessing on test_loader
    if hasattr(cfg, "DATALOADER") and hasattr(cfg.DATALOADER, "NUM_WORKERS"):
        cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    # 2) build trainer
    trainer = build_trainer(cfg)
    clip_model    = trainer.clip_model
    text_features = trainer.text_features
    clip_model.eval()

    # 3) ensure no workers in DataLoader
    try:
        old_loader = trainer.test_loader
        trainer.test_loader = DataLoader(
            old_loader.dataset,
            batch_size=old_loader.batch_size,
            shuffle=False,
            num_workers=0
        )
    except Exception:
        pass

    all_preds, all_labels, all_confs = [], [], []

    # 4) inference
    with torch.no_grad():
        for batch in trainer.test_loader:
            parsed = trainer.parse_batch_train(batch)
            images, labels = parsed[0], parsed[1]

            logits = trainer.model_inference(images)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            confs  = probs.max(dim=1).values

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    # 5) metrics
    preds_t  = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)
    confs_t  = torch.tensor(all_confs)
    acc, ece, bin_acc = compute_metrics(preds_t, labels_t, confs_t)
    print(f"[{split_name:^5}] ACC={acc:.2f}%  ECE={ece:.2f}%")

    # 6) save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"metrics_{split_name}.txt"), "w") as f:
        f.write(f"Split: {split_name}\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"ECE: {ece:.2f}%\n")

    # 7) plot
    n_bins = 20
    delta  = 1.0 / n_bins
    x_axis = np.arange(0, 1, delta)
    mids   = np.linspace(delta/2, 1-delta/2, n_bins)
    bin_arr = np.array(bin_acc)
    gap    = np.abs(mids - bin_arr)

    plt.figure(figsize=(8,8))
    plt.bar(x_axis, bin_arr, width=delta, edgecolor='k', label='Bin Acc', zorder=5)
    plt.bar(x_axis, gap, bottom=np.minimum(bin_arr, mids),
            width=delta, alpha=0.5, hatch='/', label='Gap', zorder=10)
    plt.plot([0,1], [0,1], '--', color='gray', zorder=0)
    plt.title(f"{cfg.DATASET.NAME} — {split_name}", fontsize=14)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.text(0.05, 0.95, f"ECE = {ece:.2f}%", transform=plt.gca().transAxes,
             fontsize=12, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)

    fig_path = os.path.join(output_dir, f"reliability_{split_name}.png")
    plt.tight_layout(); plt.savefig(fig_path, dpi=300); plt.close()
    print(f"[✓] Saved plot → {fig_path}\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",        type=str,   required=True,
                   help="path to your dataset root")
    p.add_argument("--config-file", type=str,   required=True,
                   help="trainer YAML (e.g. configs/trainers/CoOp/rn50.yaml)")
    p.add_argument("--datasets",    nargs="+",  required=True,
                   help="list of datasets (e.g. caltech101 food101 …)")
    p.add_argument("--splits",      nargs="+",  default=["base","new"],
                   choices=["all","base","new"],
                   help="which splits to evaluate")
    p.add_argument("--output-base", type=str,   required=True,
                   help="where to store all results")
    args = p.parse_args()

    # Loop over datasets & splits
    for ds in args.datasets:
        print(f"=== Dataset: {ds} ===")
        ds_cfg = os.path.join("configs", "datasets", f"{ds}.yaml")
        for sp in args.splits:
            out_dir = os.path.join(args.output_base, f"{ds}_{sp}")
            # build shared cfg
            cfg = get_cfg_default()
            extend_cfg(cfg)
            cfg.merge_from_file(ds_cfg)
            cfg.merge_from_file(args.config_file)
            run_one(cfg, out_dir, sp, args.root)

if __name__ == "__main__":
    main()
