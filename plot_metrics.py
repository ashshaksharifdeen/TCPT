#!/usr/bin/env python
import os
import os.path as osp
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg   # import your configuration extension function

# Import datasets so that they are registered.
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
# Metric Functions
# ----------------------------------------------------------------------------
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
                bin_confidence[bin_idx] += confidence
    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] /= bin_num_sample[idx]
            bin_confidence[idx] /= bin_num_sample[idx]
    ece_loss = 0.0
    for idx in range(num_bins):
        ece_loss += abs(bin_accuracy[idx] - bin_confidence[idx]) * bin_num_sample[idx] / len(predictions)
    return ece_loss, bin_accuracy

def compute_test_ece(trainer):
    trainer.model.eval()
    predictions, confidences, correct = [], [], []
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)
            logits = trainer.model(images)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            for i in range(len(pred)):
                predictions.append(int(pred[i].item()))
                confidences.append(conf[i].item())
                correct.append(1 if pred[i].item() == labels[i].item() else 0)
    ece, bin_acc = ECE_Loss(20, torch.tensor(predictions), torch.tensor(confidences), correct)
    return ece, bin_acc

def compute_intra_class_variance(trainer):
    trainer.model.eval()
    class_scores = {}
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)
            logits = trainer.model(images)
            #print("logits shape intra:",logits.shape) #logits shape intra: torch.Size([32, 5])
            for i, lab in enumerate(labels.cpu().tolist()):
                #print("i intra:",i.shape)
                #print("i val intra:",i)
                #print("lab intra:",lab.shape)
                #print("lab val intra:",lab)
                score = logits[i, lab]
                #print("score shape intra:",score.shape) 
                #print("score val intra:",score)
                class_scores.setdefault(lab, []).append(score)
    variances = []
    for scores in class_scores.values():
        scores_tensor = torch.stack(scores)
        var_val = torch.var(scores_tensor, unbiased=False)
        variances.append(var_val)
    return torch.stack(variances).mean().item() if variances else 0.0

def compute_inter_class_margin_std(trainer):
    trainer.model.eval()
    margins = []
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)
            logits = trainer.model(images)
            #print("logits shape inter:",logits.shape) #logits shape inter: torch.Size([32, 5])
            for i, lab in enumerate(labels.cpu().tolist()):
                #print("i inter:",i.shape)
                #print("i val inter:",i)
                #print("lab inter:",lab.shape)
                #print("lab val inter:",lab)
                s_true = logits[i, lab]
                #print("s_true inter shape:", s_true.shape)
                #print("s_true val:", s_true)
                logits_i = logits[i].clone()
                #print("logits_i inter shape:", logits_i.shape)
                logits_i[lab] = -float("inf")
                s_max = logits_i.max()
                #print("s_max inter shape:", s_max.shape)
                #print("s_max inter val:", s_max)
                margins.append(s_true - s_max)
    if margins:
        margins_tensor = torch.stack(margins)
        return torch.std(margins_tensor, unbiased=False).item()
    else:
        return 0.0

def compute_metrics(preds, labels, confs):
    correct = (preds == labels).tolist()
    ece_val, bin_acc = ECE_Loss(20, preds.tolist(), confs.tolist(), correct)
    acc = sum(correct) / len(correct)
    return acc * 100, ece_val * 100, bin_acc

# ----------------------------------------------------------------------------
# Experiment Directory Discovery
# ----------------------------------------------------------------------------
def find_experiment_dirs(base_dir):
    """
    Finds all experiment directories under base_dir that contain a checkpoint file
    named "model.pth.tar-5" in a "MultiModalPromptLearner" subfolder.
    Expected structure:
      base_dir/<CFG>_<prompt>/seedX/MultiModalPromptLearner/model.pth.tar-5
    """
    pattern = osp.join(base_dir, "**", "MultiModalPromptLearner", "model.pth.tar-5")
    files = glob.glob(pattern, recursive=True)
    exp_dirs = []
    for file in files:
        # The experiment directory is two levels up from the checkpoint file.
        exp_dir = osp.dirname(osp.dirname(file))
        if exp_dir not in exp_dirs:
            exp_dirs.append(exp_dir)
    return exp_dirs

def extract_prompt_from_exp_dir(exp_dir, cfg_prefix="vit_b16_c2_ep5_batch4_2ctx_"):
    """
    Given an experiment directory of the form
      .../MaPLe/<CFG>_<prompt>/seedX,
    extract the prompt string. It removes the known config prefix and converts underscores to spaces.
    """
    # Get the folder that contains the configuration and prompt information.
    parent = osp.dirname(exp_dir)  # e.g., .../MaPLe/<CFG>_<prompt>
    folder_name = osp.basename(parent)  # e.g., "vit_b16_c2_ep5_batch4_2ctx_a_blurry_image_of_a"
    if folder_name.startswith(cfg_prefix):
        prompt = folder_name[len(cfg_prefix):].replace("_", " ")
    else:
        prompt = folder_name.replace("_", " ")
    return prompt

# ----------------------------------------------------------------------------
# Load Trainer Using the load_model() Method (as in your reliability diagram code)
# ----------------------------------------------------------------------------
def load_trainer(model_dir, prompt_value):
    """
    Build the trainer with your configuration and load the model using trainer.load_model.
    The configuration is set to match your reliability diagram code, and we update CTX_INIT using prompt_value.
    """
    seed = 1
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.SEED = seed
    cfg.OUTPUT_DIR = model_dir
    cfg.TRAINER.NAME = "MaPLe"        # Your trainer name.
    cfg.TRAINER.MAPLE.CTX_INIT = prompt_value  # Set to the extracted prompt.
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    # Note: Use the correct dataset name (case-sensitive) if needed.
    cfg.DATASET.NAME = "OxfordFlowers"
    cfg.DATASET.ROOT = "/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    # Build the trainer using build_trainer (this calls your train.py code).
    trainer = build_trainer(cfg)
    # Load the model from the experiment directory for the given epoch.
    trainer.load_model(model_dir, epoch=5)
    trainer.model.eval()
    return trainer

# ----------------------------------------------------------------------------
# Main Function: Compute Metrics and Plot Graphs
# ----------------------------------------------------------------------------
def main():
    # Base directory for experiments.
    base_dir = "output/base2new/train_base/oxford_flowers/shots_16/MaPLe"
    experiment_dirs = find_experiment_dirs(base_dir)
    if not experiment_dirs:
        print("No experiment directories found in", base_dir)
        return

    print("Found experiment directories:")
    for exp in experiment_dirs:
        print(exp)

    results = []  # Each entry: (experiment_dir, prompt, intra_var, inter_std, ece)
    for exp_dir in experiment_dirs:
        prompt = extract_prompt_from_exp_dir(exp_dir)
        print(f"Processing experiment at {exp_dir} with prompt: '{prompt}'")
        try:
            trainer = load_trainer(exp_dir, prompt)
        except Exception as e:
            print(f"Error loading trainer from {exp_dir}: {e}")
            continue
        try:
            intra_var = compute_intra_class_variance(trainer)
            inter_std = compute_inter_class_margin_std(trainer)
            ece, bin_acc = compute_test_ece(trainer)
        except Exception as e:
            print(f"Error computing metrics for {exp_dir}: {e}")
            continue
        results.append((exp_dir, prompt, intra_var, inter_std, ece))
        print(f"[{prompt}] Intra-Class Variance: {intra_var:.4f} | Inter-Class Margin Std: {inter_std:.4f} | ECE: {ece:.4f}")

    if not results:
        print("No experiment results to plot!")
        return

    # Create directory for plots.
    plots_dir = os.path.join("output", "plots_base")
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare data arrays.
    intra_vars = np.array([r[2] for r in results])
    inter_stds = np.array([r[3] for r in results])
    ece_vals = np.array([r[4] for r in results])

    # ----- Plot 1: Intra-Class Variance vs. ECE -----
    plt.figure(figsize=(10, 8))
    plt.scatter(intra_vars, ece_vals, marker='o', color='blue', s=80)
    plt.xlabel('Intra-Class Variance')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Intra-Class Variance vs. ECE')
    coeffs = np.polyfit(intra_vars, ece_vals, 1)
    poly_eq = np.poly1d(coeffs)
    x_reg = np.linspace(np.min(intra_vars), np.max(intra_vars), 100)
    plt.plot(x_reg, poly_eq(x_reg), color='black', linestyle='--',
             label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
    plt.legend()
    intra_ece_plot_path = os.path.join(plots_dir, "intra_variance_vs_ece_base.png")
    plt.tight_layout()
    plt.savefig(intra_ece_plot_path, dpi=300)
    print(f"[✓] Saved Intra-Class Variance vs. ECE plot to {intra_ece_plot_path}")
    plt.close()

    # ----- Plot 2: Inter-Class Equivariance vs. ECE -----
    plt.figure(figsize=(10, 8))
    plt.scatter(inter_stds, ece_vals, marker='s', color='green', s=80)
    plt.xlabel('Inter-Class Margin Variability (Std. Dev.)',fontsize=18)
    plt.ylabel('Expected Calibration Error (ECE)',fontsize=18)
    plt.title('Inter-Class Margin variance vs. ECE',fontsize=20)
    # — increase tick‐label size —
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    coeffs_inter = np.polyfit(inter_stds, ece_vals, 1)
    poly_eq_inter = np.poly1d(coeffs_inter)
    x_reg_inter = np.linspace(np.min(inter_stds), np.max(inter_stds), 100)
    plt.plot(x_reg_inter, poly_eq_inter(x_reg_inter), color='black', linestyle='--',
             label=f"Fit: y={coeffs_inter[0]:.2f}x+{coeffs_inter[1]:.2f}")
    plt.legend(fontsize=14)
    inter_ece_plot_path = os.path.join(plots_dir, "flower_inter_equivariance_vs_ece_base.png")
    plt.tight_layout()
    plt.savefig(inter_ece_plot_path, dpi=300)
    print(f"[✓] Saved Inter-Class Equivariance vs. ECE plot to {inter_ece_plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
