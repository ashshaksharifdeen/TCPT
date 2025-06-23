#!/usr/bin/env python3
# margin_analysis.py

import os
import math
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

# ———————————————————————————————————————————————
# Basic Metrics
# ———————————————————————————————————————————————
def compute_acc_conf(trainer):
    """Returns (accuracy %, mean_confidence%) over test set."""
    confs, correct = [], []
    trainer.clip_model.eval()
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)[:2]
            logits = trainer.model_inference(images)
            probs  = F.softmax(logits, dim=1)
            confs.extend(probs.max(dim=1).values.cpu().tolist())
            correct.extend((probs.argmax(dim=1)==labels).cpu().tolist())
    acc_pct   = sum(correct)/len(correct)*100
    conf_pct  = np.mean(confs)*100
    return acc_pct, conf_pct

def compute_inter_std(trainer):
    """Compute inter-class margin std over test set."""
    trainer.clip_model.eval()
    margins = []
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)[:2]
            logits = trainer.model_inference(images)
            for i, lab in enumerate(labels.cpu().tolist()):
                s_true = logits[i, lab]
                row    = logits[i].clone()
                row[lab] = -float("inf")
                margins.append((s_true-row.max()).item())
    return float(torch.std(torch.tensor(margins), unbiased=False).item())

def compute_ece(trainer, num_bins=20):
    """Compute ECE (%) over test set."""
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
    total = len(preds)
    bins  = np.linspace(0,1,num_bins+1)
    ece = 0.0
    for i in range(num_bins):
        mask = [(bins[i]<c<=bins[i+1]) for c in confs]
        if any(mask):
            acc_bin  = np.mean([correct[j] for j,m in enumerate(mask) if m])
            conf_bin = np.mean([confs[j]   for j,m in enumerate(mask) if m])
            ece += abs(acc_bin-conf_bin)*sum(mask)/total
    return ece*100

# ———————————————————————————————————————————————
# Runner for one dataset+split
# ———————————————————————————————————————————————
def setup_trainer(cfg, root, split):
    cfg.defrost()
    cfg.DATASET.ROOT               = root
    cfg.DATASET.SUBSAMPLE_CLASSES  = split
    cfg.TRAINER.NAME               = "ZeroshotCLIP"
    if hasattr(cfg, "DATALOADER") and hasattr(cfg.DATALOADER, "NUM_WORKERS"):
        cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    trainer = build_trainer(cfg)
    # force single‐worker DataLoader
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

    return trainer

# ———————————————————————————————————————————————
# Experiment 1: Measured-Margin vs Over/Under-Confidence
# ———————————————————————————————————————————————
def measured_margin_experiment(base_cfg, datasets, splits, root, out_dir):
    records = []  # (label, inter_std, conf_bias)
    for ds in datasets:
        ds_yml = os.path.join("configs","datasets",f"{ds}.yaml")
        for sp in splits:
            cfg = base_cfg.clone()
            cfg.merge_from_file(ds_yml)
            trainer = setup_trainer(cfg, root, sp)

            acc, mean_conf = compute_acc_conf(trainer)
            inter_std      = compute_inter_std(trainer)
            bias           = mean_conf - acc

            label = f"{ds}_{sp}"
            records.append((label, inter_std, bias))

            print(f"[Measured] {label}: margin={inter_std:.2f}, bias={bias:.2f}%")

    # Plot
    plt.figure(figsize=(10,6))
    for label, m, bias in records:
        plt.scatter(m, bias, s=80, label=label)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Inter-Class Margin Std")
    plt.ylabel("Mean Confidence − Accuracy (%)")
    plt.title("Measured Margin vs Over/Under-Confidence")
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    f = os.path.join(out_dir, "measured_margin_bias.png")
    plt.tight_layout(); plt.savefig(f, dpi=300); plt.close()
    print(f"[✓] Saved {f}")

# ———————————————————————————————————————————————
# Experiment 2: Margin-Sweep & Optimal Range
# ———————————————————————————————————————————————
def margin_sweep_experiment(base_cfg, datasets, splits, root, out_dir):
    sweep = []  # (margin_std, ece)
    alphas = np.arange(0.5, 1.51, 0.1)

    for ds in datasets:
        ds_yml = os.path.join("configs","datasets",f"{ds}.yaml")
        for sp in splits:
            cfg = base_cfg.clone()
            cfg.merge_from_file(ds_yml)
            trainer = setup_trainer(cfg, root, sp)

            # original margin
            m0 = compute_inter_std(trainer)
            # original logit_scale parameter
            orig_log = trainer.clip_model.logit_scale.data.clone()

            for α in alphas:
                # rescale logits by α
                trainer.clip_model.logit_scale.data = orig_log + math.log(α)
                ece = compute_ece(trainer)
                sweep.append((m0*α, ece))

            # restore
            trainer.clip_model.logit_scale.data = orig_log

            print(f"[Sweep] {ds}_{sp}: m₀={m0:.2f}")

    # aggregate into bins
    all_m, all_e = np.array([s[0] for s in sweep]), np.array([s[1] for s in sweep])
    bins = np.linspace(all_m.min(), all_m.max(), 20+1)
    mids = (bins[:-1]+bins[1:]) / 2
    mean_e, std_e = [], []
    for i in range(len(mids)):
        mask = (all_m>=bins[i]) & (all_m<bins[i+1])
        if mask.any():
            mean_e.append(all_e[mask].mean())
            std_e.append(all_e[mask].std())
        else:
            mean_e.append(np.nan); std_e.append(np.nan)
    mean_e = np.array(mean_e)
    std_e  = np.array(std_e)
    idx_opt = np.nanargmin(mean_e)
    m_opt   = mids[idx_opt]

    # Plot
    plt.figure(figsize=(10,6))
    plt.scatter(all_m, all_e, color='lightgray', s=10, label='all runs')
    plt.plot(mids, mean_e, color='black', lw=2, label='mean ECE')
    plt.fill_between(mids, mean_e-std_e, mean_e+std_e,
                     color='gray', alpha=0.3, label='±1 std')
    plt.axvline(m_opt, color='red', linestyle='--',
                label=f'optimal m*={m_opt:.2f}')
    plt.xlabel("Inter-Class Margin Std")
    plt.ylabel("ECE (%)")
    plt.title("Margin Sweep: ECE vs Margin Std")
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    f = os.path.join(out_dir, "margin_sweep_optimal.png")
    plt.tight_layout(); plt.savefig(f, dpi=300); plt.close()
    print(f"[✓] Saved {f}")

# ———————————————————————————————————————————————
# Main
# ———————————————————————————————————————————————
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",        type=str,   required=True)
    p.add_argument("--config-file", type=str,   required=True)
    p.add_argument("--datasets",    nargs="+",  required=True)
    p.add_argument("--splits",      nargs="+",  default=["base","new"],
                   choices=["all","base","new"])
    p.add_argument("--output-dir",  type=str,   required=True)
    args = p.parse_args()

    # Base config
    base_cfg = get_cfg_default()
    extend_cfg(base_cfg)
    base_cfg.merge_from_file(args.config_file)

    # Run both experiments
    measured_margin_experiment(base_cfg, args.datasets, args.splits,
                               args.root, args.output_dir)
    margin_sweep_experiment(base_cfg, args.datasets, args.splits,
                            args.root, args.output_dir)

if __name__ == "__main__":
    main()
