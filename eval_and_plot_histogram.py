import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg
import trainers.maple

def ECE_Loss(num_bins, predictions, confidences, correct):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0] * num_bins
    bin_confidence = [0] * num_bins
    bin_num_sample = [0] * num_bins

    for idx in range(len(predictions)):
        confidence = confidences[idx]
        for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            if bin_lower.item() < confidence <= bin_upper.item():
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]

    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] /= bin_num_sample[idx]
            bin_confidence[idx] /= bin_num_sample[idx]

    ece_loss = 0.0
    for idx in range(num_bins):
        ece_loss += abs(bin_accuracy[idx] - bin_confidence[idx]) * bin_num_sample[idx] / len(predictions)

    return ece_loss

def compute_metrics(preds, labels, confs):
    correct = (preds == labels).tolist()
    ece_val = ECE_Loss(20, preds.tolist(), confs.tolist(), correct)
    acc = sum(correct) / len(correct)
    return acc * 100, ece_val * 100

def plot_incorrect_fraction_histogram(incorrect_confidences, save_path, title="Fraction of Incorrect Samples by Confidence"):
    import numpy as np
    import matplotlib.pyplot as plt

    # Create 10 bins from 0 to 1
    bin_edges = np.linspace(0.0, 1.0, 11)
    hist, edges = np.histogram(incorrect_confidences, bins=bin_edges)
    if len(incorrect_confidences) > 0:
        fraction = hist / len(incorrect_confidences)
    else:
        fraction = np.zeros_like(hist)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.bar(edges[:-1], fraction, width=np.diff(edges), align='edge', color='skyblue', edgecolor='black')
    ax.set_title(title, fontsize=20, weight='bold')
    ax.set_xlabel("Confidence", fontsize=30, weight='bold')
    ax.set_ylabel("Fraction of Incorrect Samples", fontsize=30, weight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 0.35])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    print(f"[✓] Saved histogram to {save_path}")
    plt.close()

def main(args):
    seed = 1
    prompt = "a photo of a"

    # Build configuration and load the model
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
    trainer.load_model(os.path.join(args.model_dir, f"seed{seed}"), epoch=args.load_epoch)
    trainer.model.eval()

    all_preds = []
    all_labels = []
    all_confs = []
    image_feats_list = []

    # Process the entire test set
    with torch.no_grad():
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_train(batch)
            logits = trainer.model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            max_conf = probs.max(dim=1).values

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            all_confs.extend(max_conf.cpu().tolist())
            image_feats_list.append(trainer.model.imfeatures.cpu())

    # Compute overall metrics for the test set
    preds_t = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)
    confs_t = torch.tensor(all_confs)
    acc, ece = compute_metrics(preds_t, labels_t, confs_t)
    print(f"[Metrics] ACC={acc:.2f}%  ECE={ece:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "alignment_metrics.txt"), "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Accuracy: {acc:.2f}%\nECE: {ece:.2f}%\n")

    # Plot histogram for fraction of incorrect samples by confidence
    incorrect_confidences = [conf for conf, p, l in zip(all_confs, all_preds, all_labels) if p != l]
    hist_save_path = os.path.join(args.output_dir, "incorrect_confidence_histogram.png")
    plot_incorrect_fraction_histogram(incorrect_confidences, hist_save_path,
                                      title="Fraction of Incorrect Samples by Confidence")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--load-epoch", type=int, default=5)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--subsample-classes", type=str, default="new", help="Class split: base/new/all")
    args = parser.parse_args()
    main(args)
