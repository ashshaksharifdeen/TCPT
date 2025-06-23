#!/usr/bin/env python3
# zero_inter_intra.py

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
import trainers.zsclip  # registers ZeroshotCLIP

def compute_acc_ece(trainer):
    preds, confs, correct = [], [], []
    trainer.clip_model.eval()
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)[:2]
            logits = trainer.model_inference(images)
            probs  = F.softmax(logits, dim=1)
            preds.extend(probs.argmax(dim=1).cpu().tolist())
            confs.extend(probs.max(dim=1).values.cpu().tolist())
            correct.extend((probs.argmax(dim=1)==labels).cpu().tolist())
    acc = sum(correct) / len(correct) * 100

    # ECE
    num_bins = 20
    bins = torch.linspace(0, 1, num_bins+1)
    lowers, uppers = bins[:-1], bins[1:]
    bin_acc, bin_conf, bin_cnt = [0]*num_bins, [0]*num_bins, [0]*num_bins
    for cfd, corr in zip(confs, correct):
        for b, (l, u) in enumerate(zip(lowers, uppers)):
            if l < cfd <= u:
                bin_cnt[b]  += 1
                bin_acc[b]  += corr
                bin_conf[b] += cfd
    ece = 0.0
    for b in range(num_bins):
        if bin_cnt[b]:
            bin_acc[b]  /= bin_cnt[b]
            bin_conf[b] /= bin_cnt[b]
            ece += abs(bin_acc[b] - bin_conf[b]) * (bin_cnt[b]/len(correct))
    return acc, ece * 100

def compute_intra_and_inter(trainer):
    trainer.clip_model.eval()
    class_scores, margins = {}, []
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)[:2]
            logits = trainer.model_inference(images)
            for i, lab in enumerate(labels.cpu().tolist()):
                s_true = logits[i, lab]
                class_scores.setdefault(lab, []).append(s_true)
                row = logits[i].clone()
                row[lab] = -float("inf")
                margins.append((s_true - row.max()).item())
    variances = [torch.var(torch.stack(v), unbiased=False).item()
                 for v in class_scores.values()]
    intra_var = float(np.mean(variances)) if variances else 0.0
    inter_std = float(np.std(np.array(margins), ddof=0)) if margins else 0.0
    return intra_var, inter_std

def run_one(cfg, dataset, split, root):
    cfg.defrost()
    cfg.DATASET.ROOT              = root
    cfg.DATASET.SUBSAMPLE_CLASSES = split
    cfg.TRAINER.NAME              = "ZeroshotCLIP"
    if hasattr(cfg, "DATALOADER") and hasattr(cfg.DATALOADER, "NUM_WORKERS"):
        cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    trainer = build_trainer(cfg)
    try:
        old = trainer.test_loader
        trainer.test_loader = DataLoader(
            old.dataset,
            batch_size=old.batch_size,
            shuffle=False,
            num_workers=0
        )
    except:
        pass

    acc, ece               = compute_acc_ece(trainer)
    intra_var, inter_std   = compute_intra_and_inter(trainer)

    print(f"{dataset:15} | {split:5} | ACC={acc:6.2f}% | ECE={ece:6.2f}% | "
          f"IntraVar={intra_var:6.4f} | InterStd={inter_std:6.4f}")

    return acc, ece, intra_var, inter_std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",        type=str,   required=True)
    parser.add_argument("--config-file", type=str,   required=True)
    parser.add_argument("--datasets",    nargs="+",  required=True)
    parser.add_argument("--splits",      nargs="+",  default=["base","new"],
                        choices=["all","base","new"])
    parser.add_argument("--output-dir",  type=str,   required=True)
    args = parser.parse_args()

    base_cfg = get_cfg_default()
    extend_cfg(base_cfg)
    base_cfg.merge_from_file(args.config_file)

    summary = []  # list of (label, intra_var, inter_std, ece)
    for ds in args.datasets:
        ds_cfg = os.path.join("configs", "datasets", f"{ds}.yaml")
        for sp in args.splits:
            cfg = base_cfg.clone()
            cfg.merge_from_file(ds_cfg)
            _, ece, intra, inter = run_one(cfg, ds, sp, args.root)
            summary.append((f"{ds}_{sp}", intra, inter, ece))

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate a distinct color for each point
    cmap = plt.get_cmap('tab20')
    num = len(summary)
    colors = [cmap(i % cmap.N) for i in range(num)]

    # 1) Intra‐Class Variance vs. ECE
    plt.figure(figsize=(10,6))
    for (label, intra, _, ece), color in zip(summary, colors):
        plt.scatter(intra, ece, color=color, s=80, label=label)
    plt.xlabel("Intra-Class Variance")
    plt.ylabel("ECE (%)")
    plt.title("Intra-Class Variance vs. ECE")
    plt.grid(alpha=0.3)
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    p1 = os.path.join(args.output_dir, "intra_variance_vs_ece_summary.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[✓] Saved {p1}")

    # 2) Inter‐Class Margin Std vs. ECE
    plt.figure(figsize=(10,6))
    for (label, _, inter, ece), color in zip(summary, colors):
        plt.scatter(inter, ece, marker='s', color=color, s=80, label=label)
    plt.xlabel("Inter-Class Margin Std")
    plt.ylabel("ECE (%)")
    plt.title("Inter-Class Margin Std vs. ECE")
    plt.grid(alpha=0.3)
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    p2 = os.path.join(args.output_dir, "inter_margin_std_vs_ece_summary.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[✓] Saved {p2}")

if __name__ == "__main__":
    main()
