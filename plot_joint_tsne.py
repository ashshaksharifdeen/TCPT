import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE

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

def select_one_sample_per_class(trainer):
    """
    Iterates over the test loader and selects the first image sample 
    encountered for each class.
    
    Returns:
      - image_feats: Tensor of shape (num_classes, 512)
      - gt_labels: List of ground-truth labels for the selected images.
      - pred_labels: List of predicted class for each image.
      - confs: List of confidence values for each image.
    """
    selected = {}  # mapping from class label to (img_feature, pred, conf)
    with torch.no_grad():
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_train(batch)
            logits = trainer.model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            max_conf = probs.max(dim=1).values
            for i, lab in enumerate(label.cpu().tolist()):
                if lab not in selected:
                    selected[lab] = (trainer.model.imfeatures[i].cpu(), pred[i].item(), max_conf[i].item())
            num_classes = trainer.model.textfeatures.size(0)
            if len(selected) >= num_classes:
                break
    sorted_keys = sorted(selected.keys())
    image_feats = []
    gt_labels = []
    pred_labels = []
    confs = []
    for k in sorted_keys:
        feat, p, c = selected[k]
        image_feats.append(feat)
        gt_labels.append(k)
        pred_labels.append(p)
        confs.append(c)
    image_feats = torch.stack(image_feats)
    return image_feats, gt_labels, pred_labels, confs

def plot_joint_tsne(selected_img_feats, gt_labels, pred_labels, text_feats, save_path, title="One Image per Class + Text Embeddings"):
    """
    Plots joint t-SNE for one image sample per class combined with text embeddings.
    
    Args:
      selected_img_feats: Tensor of shape (num_classes, 512) for one image per class.
      gt_labels: List of ground-truth labels (one per class).
      pred_labels: List of predicted labels for the selected images.
      text_feats: Tensor of shape (num_classes, 512) for text embeddings.
    """
    import matplotlib.cm as cm

    # Normalize features
    selected_img_feats = F.normalize(selected_img_feats.float(), dim=1)
    text_feats = F.normalize(text_feats.float(), dim=1)

    # Combine features: first part: one image per class; second: text embeddings
    all_feats = torch.cat([selected_img_feats, text_feats], dim=0)  # shape (C+1, 512)
    all_feats_np = all_feats.numpy()

    # For coloring: image features use ground-truth labels; text features use class indices (0 to C-1)
    text_cls = list(range(text_feats.size(0)))
    all_labels = gt_labels + text_cls
    all_types = ["img"] * selected_img_feats.size(0) + ["txt"] * text_feats.size(0)

    perplexity_val = min(30, all_feats_np.shape[0]-1)
    tsne_out = TSNE(n_components=2, random_state=42, perplexity=perplexity_val).fit_transform(all_feats_np)

    plt.figure(figsize=(10, 8))
    cmap = cm.get_cmap("tab20", 20)

    # Plot image features (first C points)
    for i in range(selected_img_feats.size(0)):
        marker = "o"
        color = cmap(all_labels[i] % 20)
        edgecolor = "red" if pred_labels[i] != gt_labels[i] else "none"
        plt.scatter(tsne_out[i, 0], tsne_out[i, 1], marker=marker, color=color,
                    s=100, alpha=0.8, edgecolors=edgecolor, linewidths=1)

    # Plot text embeddings (remaining points)
    offset = selected_img_feats.size(0)
    for j in range(text_feats.size(0)):
        marker = "^"
        color = cmap(j % 20)
        plt.scatter(tsne_out[offset+j, 0], tsne_out[offset+j, 1], marker=marker,
                    color=color, s=120, alpha=0.9)

    # --- Draw arrows ---
    # Always draw a green arrow from each image to its ground-truth text embedding
    for i, gt_lbl in enumerate(gt_labels):
        x_img, y_img = tsne_out[i]
        x_gt, y_gt = tsne_out[offset + gt_lbl]  # ground-truth text embedding
        plt.arrow(x_img, y_img, x_gt - x_img, y_gt - y_img,
                  color="green", alpha=0.5, width=0.002, head_width=0.03)
        mx_gt, my_gt = (x_img + x_gt)/2, (y_img + y_gt)/2
        plt.text(mx_gt, my_gt, f"GT: {gt_lbl}", fontsize=8, color="green")

    # If mispredicted, also draw a red arrow from image to predicted text embedding
    for i, (gt_lbl, pred_lbl) in enumerate(zip(gt_labels, pred_labels)):
        if pred_lbl != gt_lbl:
            x_img, y_img = tsne_out[i]
            x_pred, y_pred = tsne_out[offset + pred_lbl]
            plt.arrow(x_img, y_img, x_pred - x_img, y_pred - y_img,
                      color="red", alpha=0.5, width=0.002, head_width=0.03)
            mx_pred, my_pred = (x_img + x_pred)/2, (y_img + y_pred)/2
            plt.text(mx_pred, my_pred, f"Pred: {pred_lbl}", fontsize=8, color="red")

    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✓] Saved joint t-SNE plot to {save_path}")
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

    # Select one image sample per class from the test set.
    selected_img_feats, gt_labels, pred_labels, confs = select_one_sample_per_class(trainer)
    
    # Compute metrics on the selected samples (optional)
    preds_tensor = torch.tensor(pred_labels)
    gt_tensor = torch.tensor(gt_labels)
    confs_tensor = torch.tensor(confs)
    acc, ece = compute_metrics(preds_tensor, gt_tensor, confs_tensor)
    print(f"[Selected Samples] ACC={acc:.2f}%  ECE={ece:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "selected_sample_metrics.txt"), "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Accuracy on selected samples: {acc:.2f}%\nECE on selected samples: {ece:.2f}%\n")

    # Get text features (one per class)
    text_feats = trainer.model.textfeatures.cpu().float()

    dataset_name = os.path.splitext(os.path.basename(args.dataset_config_file))[0]
    save_path = os.path.join(args.output_dir, f"{dataset_name}_joint_tsne_one_per_class.png")
    plot_joint_tsne(selected_img_feats, gt_labels, pred_labels, text_feats, save_path,
                      title=f"One Image per Class + Text Embeddings - {dataset_name}")

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
