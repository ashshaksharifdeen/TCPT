import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg
import trainers.maple

# Import GradCAM utilities if needed (not used in the t-SNE plotting below)
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

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

def select_mispredicted_sample(trainer, threshold=0.7):
    """
    Iterates over the test loader and selects the first image sample that is mispredicted
    (predicted label != ground truth) and whose predicted confidence is greater than the given threshold.
    Returns:
      - img_feat: Tensor of shape (512,) for the image feature.
      - gt_label: Ground-truth label (int).
      - pred_label: Predicted label (int).
      - pred_conf: Predicted confidence (float).
      - gt_conf: Ground-truth confidence (float) computed from the probability of the ground-truth class.
    """
    with torch.no_grad():
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_train(batch)
            logits = trainer.model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            max_conf = probs.max(dim=1).values
            for i, (gt, p) in enumerate(zip(label.cpu().tolist(), pred.cpu().tolist())):
                # Compute ground-truth confidence:
                gt_conf = probs[i, gt].item()
                pred_conf = max_conf[i].item()
                if (p != gt) and (pred_conf > threshold):
                    img_feat = trainer.model.imfeatures[i].cpu()
                    return img_feat, gt, p, pred_conf, gt_conf
    return None, None, None, None, None

def plot_joint_tsne_mispredicted(img_feat, gt_label, pred_label, pred_conf, gt_conf, text_feats, save_path,
                                 title="Mispredicted Sample + Text Embeddings"):
    """
    Creates a joint t-SNE plot for one selected mispredicted sample and the text embeddings.
    It draws:
      - A green arrow from the image feature to the ground-truth text embedding, annotated with the ground-truth confidence.
      - A red arrow from the image feature to the predicted text embedding, annotated with the predicted confidence.
    
    Args:
      img_feat: Tensor of shape (512,) for the image feature.
      gt_label: Ground-truth class label (int).
      pred_label: Predicted class label (int).
      pred_conf: Predicted confidence (float).
      gt_conf: Ground-truth confidence (float).
      text_feats: Tensor of shape (num_classes, 512) for text embeddings.
      save_path: Path to save the t-SNE plot.
      title: Title for the plot.
    """
    import matplotlib.cm as cm

    # Normalize features
    img_feat = F.normalize(img_feat.unsqueeze(0).float(), dim=1)  # shape (1, 512)
    text_feats = F.normalize(text_feats.float(), dim=1)  # shape (C, 512)

    # Combine features: first the image feature, then the text embeddings.
    all_feats = torch.cat([img_feat, text_feats], dim=0).numpy()
    
    # For coloring: image gets the ground-truth label; text embeddings get class indices.
    all_labels = [gt_label] + list(range(text_feats.size(0)))
    all_types = ["img"] + ["txt"] * text_feats.size(0)
    
    perplexity_val = min(30, all_feats.shape[0] - 1)
    tsne_out = TSNE(n_components=2, random_state=42, perplexity=perplexity_val).fit_transform(all_feats)
    
    plt.figure(figsize=(10, 8))
    cmap = cm.get_cmap("tab20", 20)
    
    # Plot image feature.
    marker = "o"
    color = cmap(all_labels[0] % 20)
    plt.scatter(tsne_out[0, 0], tsne_out[0, 1], marker=marker, color=color, s=120, alpha=0.9)
    
    # Plot text embeddings.
    offset = 1
    for j in range(text_feats.size(0)):
        marker = "^"
        color = cmap(j % 20)
        plt.scatter(tsne_out[offset+j, 0], tsne_out[offset+j, 1], marker=marker,
                    color=color, s=100, alpha=0.9)
    
    # Draw green arrow: image -> ground-truth text embedding.
    x_img, y_img = tsne_out[0]
    x_gt, y_gt = tsne_out[offset + gt_label]  # ground-truth text embedding location
    plt.arrow(x_img, y_img, x_gt - x_img, y_gt - y_img, color="green",
              alpha=0.5, width=0.002, head_width=0.03)
    mx_gt, my_gt = (x_img + x_gt) / 2, (y_img + y_gt) / 2
    plt.text(mx_gt, my_gt, f"GT: {gt_label}\nConf: {gt_conf*100:.1f}%", fontsize=8, color="green")
    
    # Draw red arrow: image -> predicted text embedding.
    x_pred, y_pred = tsne_out[offset + pred_label]
    plt.arrow(x_img, y_img, x_pred - x_img, y_pred - y_img, color="red",
              alpha=0.5, width=0.002, head_width=0.03)
    mx_pred, my_pred = (x_img + x_pred) / 2, (y_img + y_pred) / 2
    plt.text(mx_pred, my_pred, f"Pred: {pred_label}\nConf: {pred_conf*100:.1f}%", fontsize=8, color="red")
    
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✓] Saved joint t-SNE plot to {save_path}")
    plt.close()

def main(args):
    seed = 1
    prompt = "a photo of a"
    
    # Build configuration and load the model.
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
    
    # Select one mispredicted sample with predicted confidence > threshold.
    img_feat, gt_label, pred_label, pred_conf, gt_conf = select_mispredicted_sample(trainer, threshold=0.7)
    if img_feat is None:
        print("No mispredicted sample with confidence > 70% was found.")
        return
    print(f"[Selected Sample] GT: {gt_label}, Pred: {pred_label}, Pred Conf: {pred_conf*100:.1f}%, GT Conf: {gt_conf*100:.1f}%")
    
    # Get text features (one per class) from the model.
    text_feats = trainer.model.textfeatures.cpu().float()
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_config_file))[0]
    save_path = os.path.join(args.output_dir, f"{dataset_name}_joint_tsne_mispredicted.png")
    plot_joint_tsne_mispredicted(img_feat, gt_label, pred_label, pred_conf, gt_conf, text_feats, save_path,
                                 title=f"Mispredicted Sample + Text Embeddings - {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--config-file", type=str, required=True, help="Trainer config YAML")
    parser.add_argument("--dataset-config-file", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--model-dir", type=str, required=True, help="Base dir with seed subfolders")
    parser.add_argument("--load-epoch", type=int, default=5, help="Epoch to load")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the t-SNE plot")
    parser.add_argument("--subsample-classes", type=str, default="new", help="Class split: base/new/all")
    args = parser.parse_args()
    main(args)
