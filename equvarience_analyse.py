#!/usr/bin/env python
import os
import os.path as osp
import argparse
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.cuda.amp import GradScaler, autocast

# Import from Dassl modules.
from dassl.engine import TRAINER_REGISTRY, TrainerX, build_trainer
from dassl.metrics import compute_accuracy
from dassl.utils import (load_pretrained_weights, load_checkpoint, setup_logger,
                         set_random_seed, collect_env_info)
from dassl.optim import build_optimizer, build_lr_scheduler

# Import CfgNode from yacs and default config from Dassl.
from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default

# Import CLIP and its tokenizer.
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

# Custom dataset imports (for registration)
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

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt

####################################
# Configuration Extensions
####################################
def extend_cfg(cfg):
    """
    Extend the configuration.
    (Epoch count, etc., are defined in the YAML config file.)
    """
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"
    cfg.TRAINER.IVLP.PREC = "fp16"
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT.PREC = "fp16"
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

####################################
# Calibration Metric Functions
####################################
def ECE_Loss(num_bins, predictions, confidences, correct):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0] * num_bins
    bin_confidence = [0] * num_bins
    bin_num_sample = [0] * num_bins

    for idx in range(len(predictions)):
        for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            if bin_lower.item() < confidences[idx] <= bin_upper.item():
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
    return ECE_Loss(20, predictions, confidences, correct)

def compute_intra_class_variance(trainer):
    trainer.model.eval()
    class_scores = {}
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)
            logits = trainer.model(images)
            for i, lab in enumerate(labels.cpu().tolist()):
                score = logits[i, lab]
                class_scores.setdefault(lab, []).append(score)
    variances = [torch.var(torch.stack(scores), unbiased=False) for scores in class_scores.values()]
    return torch.stack(variances).mean().item() if variances else 0.0

def compute_inter_class_margin_std(trainer):
    trainer.model.eval()
    margins = []
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_train(batch)
            logits = trainer.model(images)
            for i, lab in enumerate(labels.cpu().tolist()):
                s_true = logits[i, lab]
                logits_i = logits[i].clone()
                logits_i[lab] = -float("inf")
                s_max = logits_i.max()
                margins.append(s_true - s_max)
    return torch.std(torch.stack(margins), unbiased=False).item() if margins else 0.0

####################################
# Helper Function to Update Prompt
####################################
def update_model_prompt(trainer, prompt):
    """
    Update the prompt in the configuration and use the stored CLIP model (from the prompt learner)
    to re-tokenize and update the prompt embeddings.
    """
    print(f"\n[Update] Changing prompt to: '{prompt}'")
    trainer.cfg.defrost()
    trainer.cfg.TRAINER.MAPLE.CTX_INIT = prompt
    trainer.cfg.freeze()

    classnames = trainer.dm.dataset.classnames
    new_prompts = [f"{prompt} {name.replace('_', ' ')}." for name in classnames]
    new_tokenized_prompts = torch.cat([clip.tokenize(p) for p in new_prompts]).to(trainer.device)
    trainer.model.prompt_learner.tokenized_prompts = new_tokenized_prompts
    # Use the stored CLIP model from the prompt learner to compute embeddings
    new_embedding = trainer.model.prompt_learner.clip_model.token_embedding(new_tokenized_prompts)
    new_embedding = new_embedding.to(trainer.device).type(trainer.model.text_encoder.dtype)
    n_ctx = trainer.model.prompt_learner.n_ctx
    trainer.model.prompt_learner.token_prefix = new_embedding[:, :1, :]
    trainer.model.prompt_learner.token_suffix = new_embedding[:, 1 + n_ctx:, :]
    print("-> Updated tokenized prompts and embeddings based on the new prompt.")

####################################
# Main Evaluation Function
####################################
def main(args):
    from dassl.config import get_cfg_default
    from dassl.utils import setup_logger, set_random_seed, collect_env_info

    # Build configuration.
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed >= 0:
        cfg.SEED = args.seed
    if args.subsample_classes:
        cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample_classes
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.trainer:
        cfg.defrost()
        cfg.TRAINER.NAME = args.trainer
        cfg.freeze()

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)
    print("Collecting env info ...")
    print("** System Info **\n{}\n".format(collect_env_info()))

    # List of prompt variants.
    prompts = [
        "a photo of a",
        "an image of a",
        "a picture of a",
        "a close-up of a",
        "a detailed photo of a",
        "a high resolution image of a",
        "a blurry photo of a",
        "a stunning photo of a",
        "a low-light image of a",
        "a vibrant photo of a",
        "a minimalist depiction of a",
        "a highly detailed drawing of a",
        "a realistic photo of a",
        "a surreal image of a",
        "a painted picture of a",
        "a sketch of a",
        "a cartoon illustration of a",
        "a digital art piece of a",
        "a surrealistic photo of a",
        "an artistic portrayal of a"
    ]
    results = []  # To store (prompt, intra_var, inter_margin_std, ECE)

    # For each prompt variant, create a fresh trainer instance and train using its built-in loop.
    for idx, prompt in enumerate(prompts):
        print(f"\n====== Training for Prompt Variant {idx+1}/{len(prompts)}: '{prompt}' ======")
        current_trainer = build_trainer(cfg)  # Fresh trainer instance per prompt variant
        update_model_prompt(current_trainer, prompt)
        variant_output_dir = osp.join(cfg.OUTPUT_DIR, f"prompt_{idx+1}")
        current_trainer.cfg.defrost()
        current_trainer.cfg.OUTPUT_DIR = variant_output_dir
        current_trainer.cfg.freeze()
        os.makedirs(variant_output_dir, exist_ok=True)
        
        print(f"--- Training using prompt variant '{prompt}' ---")
        # Call the trainer's training routine. (It is assumed that the trainer's .train() loop takes care of
        # iterating over batches; each batch's forward pass creates a fresh graph that is freed after .backward().)
        current_trainer.train()

        # After training for this prompt, compute calibration metrics.
        intra_var = compute_intra_class_variance(current_trainer)
        inter_margin_std = compute_inter_class_margin_std(current_trainer)
        ece_val = compute_test_ece(current_trainer)
        print(f"Prompt: '{prompt}' => Intra-Class Var: {intra_var:.4f}, Inter-Class Margin Std: {inter_margin_std:.4f}, ECE: {ece_val:.4f}")
        results.append((prompt, intra_var, inter_margin_std, ece_val))

    # Save the results to a text file.
    eval_output_dir = osp.join(cfg.OUTPUT_DIR, "prompt_calibration")
    os.makedirs(eval_output_dir, exist_ok=True)
    results_path = osp.join(eval_output_dir, "prompt_calibration_results.txt")
    with open(results_path, "w") as f:
        for res in results:
            f.write(f"Prompt: {res[0]}\tIntraVar: {res[1]:.4f}\tInterMarginStd: {res[2]:.4f}\tECE: {res[3]:.4f}\n")
    print(f"\n[✓] Saved prompt calibration results to {results_path}")

    # Create a separate directory to save plots.
    plots_dir = osp.join(cfg.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    intra_vars = np.array([r[1] for r in results])
    inter_stds = np.array([r[2] for r in results])
    ece_vals = np.array([r[3] for r in results])

    # Plot: Intra-Class Variance vs. ECE.
    plt.figure(figsize=(8, 6))
    plt.scatter(intra_vars, ece_vals, marker='o', color='blue', s=80)
    plt.xlabel('Intra-Class Variance')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Intra-Class Variance vs. ECE')
    coeffs = np.polyfit(intra_vars, ece_vals, 1)
    poly_eq = np.poly1d(coeffs)
    x_reg = np.linspace(np.min(intra_vars), np.max(intra_vars), 100)
    plt.plot(x_reg, poly_eq(x_reg), color='black', linestyle='--', label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
    plt.legend()
    intra_ece_plot_path = osp.join(plots_dir, "intra_variance_vs_ece.png")
    plt.tight_layout()
    plt.savefig(intra_ece_plot_path, dpi=300)
    print(f"[✓] Saved Intra-Class Variance vs. ECE plot to {intra_ece_plot_path}")
    plt.close()

    # Plot: Inter-Class Equivariance vs. ECE.
    plt.figure(figsize=(8, 6))
    plt.scatter(inter_stds, ece_vals, marker='s', color='green', s=80)
    plt.xlabel('Inter-Class Margin Variability (Std. Dev.)')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Inter-Class Equivariance vs. ECE')
    coeffs_inter = np.polyfit(inter_stds, ece_vals, 1)
    poly_eq_inter = np.poly1d(coeffs_inter)
    x_reg_inter = np.linspace(np.min(inter_stds), np.max(inter_stds), 100)
    plt.plot(x_reg_inter, poly_eq_inter(x_reg_inter), color='black', linestyle='--', label=f"Fit: y={coeffs_inter[0]:.2f}x+{coeffs_inter[1]:.2f}")
    plt.legend()
    inter_ece_plot_path = osp.join(plots_dir, "inter_equivariance_vs_ece.png")
    plt.tight_layout()
    plt.savefig(inter_ece_plot_path, dpi=300)
    print(f"[✓] Saved Inter-Class Equivariance vs. ECE plot to {inter_ece_plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset root")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (for resume)")
    parser.add_argument("--seed", type=int, default=-1, help="seed; positive value fixes random seed")
    parser.add_argument("--config-file", type=str, default="", help="path to model config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to dataset config file")
    parser.add_argument("--trainer", type=str, default="MaPLe", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="directory of trained model")
    parser.add_argument("--load-epoch", type=int, default=5, help="checkpoint epoch to load")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--subsample-classes", type=str, default="new", help="class split: base/new/all")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
