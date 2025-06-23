#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg
import trainers.maple

def ECE_Loss(num_bins, predictions, confidences, correct):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0.0] * num_bins
    bin_confidence = [0.0] * num_bins
    bin_num_sample = [0] * num_bins

    for idx in range(len(predictions)):
        conf = confidences[idx]
        for b, (low, high) in enumerate(zip(bin_lowers, bin_uppers)):
            if low.item() < conf <= high.item():
                bin_num_sample[b] += 1
                bin_accuracy[b] += correct[idx]
                bin_confidence[b] += conf

    for b in range(num_bins):
        if bin_num_sample[b] > 0:
            bin_accuracy[b] /= bin_num_sample[b]
            bin_confidence[b] /= bin_num_sample[b]

    ece = 0.0
    total = len(predictions)
    for b in range(num_bins):
        ece += abs(bin_accuracy[b] - bin_confidence[b]) * (bin_num_sample[b] / total)

    return ece, bin_accuracy

def compute_metrics(preds, labels, confs):
    correct = (preds == labels).tolist()
    ece_val, bin_acc = ECE_Loss(20, preds.tolist(), confs.tolist(), correct)
    acc = sum(correct) / len(correct)
    return acc * 100, ece_val * 100, bin_acc

def plot_incorrect_fraction_histogram(incorrect_confidences, save_path,
                                      title="Fraction of Incorrect Samples by Confidence"):
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(incorrect_confidences, bins=edges)
    frac = hist / len(incorrect_confidences) if len(incorrect_confidences) > 0 else np.zeros_like(hist)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.bar(edges[:-1], frac,
           width=np.diff(edges), align='edge',
           color='skyblue', edgecolor='black', zorder=5)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Confidence", fontsize=18)
    ax.set_ylabel("Fraction of Incorrect Samples", fontsize=18)
    # start x-axis at 0.2 instead of 0
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.0, 0.35)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(alpha=0.3, linestyle='--', zorder=0)

    plt.tight_layout()
    fig.savefig(save_path, format="png", dpi=300)
    print(f"[✓] Saved histogram to {save_path}")
    plt.close(fig)

def main(args):
    seed = 2
    prompt = "a photo of a"

    # Build & load model
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.SEED = seed
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.CTX_INIT = prompt
    cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample_classes
    cfg.freeze()

    trainer = build_trainer(cfg)
    trainer.load_model(os.path.join(args.model_dir, f"seed{seed}"),
                       epoch=args.load_epoch)
    trainer.model.eval()

    # Gather preds, labels, confs
    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_train(batch)
            logits = trainer.model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())

    preds_t = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)
    confs_t = torch.tensor(all_confs)
    acc, ece, bin_acc = compute_metrics(preds_t, labels_t, confs_t)
    print(f"[Metrics] ACC={acc:.2f}%  ECE={ece:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "alignment_metrics.txt"), "w") as f:
        f.write(f"Prompt: {prompt}\nAccuracy: {acc:.2f}%\nECE: {ece:.2f}%\n")
    print(f"[✓] Saved metrics to {args.output_dir}/alignment_metrics.txt")

    # Histogram of incorrect confidence
    incorrect_conf = [c for c, p, l in zip(all_confs, all_preds, all_labels) if p != l]
    hist_path = os.path.join(args.output_dir, "incorrect_confidence_histogram.png")
    plot_incorrect_fraction_histogram(incorrect_conf, hist_path)

    # Reliability diagram
    n_bins = 20
    delta = 1.0 / n_bins
    x = np.arange(0, 1, delta)
    mid = np.linspace(delta/2, 1 - delta/2, n_bins)
    bin_acc_arr = np.array(bin_acc)
    gap = np.abs(mid - bin_acc_arr)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.bar(x, bin_acc_arr,
           width=delta, align="edge",
           color="b", edgecolor="k",
           label="Outputs", zorder=5)
    ax.bar(x, gap,
           bottom=np.minimum(bin_acc_arr, mid),
           width=delta, align="edge",
           color="mistyrose", alpha=0.5,
           edgecolor="r", hatch="/",
           label="Gap", zorder=10)
    ax.plot([0, 1], [0, 1],
            linestyle="--", color="gray",
            zorder=15)

    # Put ECE into the legend title
    ax.legend(
        loc="upper left",
        fontsize=14,
        framealpha=1.0,
        title=f"ECE = {ece:.2f}%",
        title_fontsize=14
    )

    # Styling
    # start x-axis at 0.2 instead of 0
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 1)
    ax.grid(color="gray", linestyle=(0, (1, 5)), linewidth=1, zorder=0)
    #ax.set_title("Reliability Diagram MAPLE() - Base", fontsize=20)
    ax.set_xlabel("Confidence", fontsize=18)
    ax.set_ylabel("Accuracy", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()
    reliability_save_path = os.path.join(
        args.output_dir,
        f"Reliability_diagram-HT-adapt-rn_{args.subsample_classes}_"
        f"{os.path.splitext(os.path.basename(args.config_file))[0]}.png"
    )
    fig.savefig(reliability_save_path, dpi=300)
    print(f"[✓] Saved reliability diagram to {reliability_save_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--load-epoch", type=int, default=5)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--subsample-classes", type=str, default="new",
                        help="Class split: base/new/all")
    args = parser.parse_args()
    main(args)
