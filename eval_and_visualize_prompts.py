import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import setup_logger
from train import extend_cfg
import trainers.maple

# === ECE and Accuracy ===
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

    return ece_loss, bin_accuracy, bin_confidence, bin_num_sample


def compute_metrics(result_dict):
    preds = torch.tensor(result_dict['prediction']).int()
    labels = torch.tensor(result_dict['label']).int()
    confidences = torch.tensor(result_dict['max_confidence'])

    correct = (preds == labels).tolist()
    ece_data = ECE_Loss(20, result_dict['prediction'], result_dict['max_confidence'], correct)
    acc = sum(correct) / len(correct)
    return acc * 100, ece_data[0] * 100


# === t-SNE plot ===
def plot_tsne(tsne_feats, labels, title, save_path):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))  # Wider figure for space
    unique_labels = sorted(list(set(labels)))

    # Use a tab10 colormap with as many colors as unique labels
    colors = cm.get_cmap("tab10", len(unique_labels))

    # Plot each label's points
    for i, label in enumerate(unique_labels):
        idxs = [j for j, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            tsne_feats[idxs, 0],
            tsne_feats[idxs, 1],
            label=label,
            alpha=0.7,
            s=35,
            color=colors(i)
        )

    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Put legend on the right side
    # 'loc' is the anchor point on the legend,
    # 'bbox_to_anchor' is the anchor on the axes
    plt.legend(
        title="Prompt Style",
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)  # Slightly outside the main plot
    )

    # This ensures everything fits well, including the legend
    plt.tight_layout(rect=[0, 0, 0.8, 1])  
    plt.savefig(save_path, dpi=300)
    print(f"[✓] Saved: {save_path}")
    plt.close()



# === Main evaluation + t-SNE ===
def evaluate_and_visualize(prompt_styles, seeds, args):
    all_image_feats, all_text_feats, all_labels = [], [], []
    metrics_result = {}

    for seed in seeds:
        for style in prompt_styles:
            cfg = get_cfg_default()
            extend_cfg(cfg)
            cfg.merge_from_file(args.dataset_config_file)
            cfg.merge_from_file(args.config_file)
            cfg.defrost()
            cfg.SEED = seed
            cfg.TRAINER.NAME = "MaPLe"
            cfg.TRAINER.MAPLE.CTX_INIT = style
            cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample_classes
            cfg.freeze()

            trainer = build_trainer(cfg)
            seed_dir = os.path.join(args.model_dir, f"seed{seed}")
            trainer.load_model(seed_dir, epoch=args.load_epoch)
            trainer.model.eval()
            loader = trainer.test_loader

            result_dict = {'prediction': [], 'label': [], 'max_confidence': []}
            image_features_list, text_features_list = [], []

            with torch.no_grad():
                for batch in loader:
                    image, label = trainer.parse_batch_train(batch)
                    logits = trainer.model(image) #label
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax(dim=1)
                    max_conf = probs.max(dim=1).values

                    result_dict['prediction'].extend(pred.cpu().tolist())
                    result_dict['label'].extend(label.cpu().tolist())
                    result_dict['max_confidence'].extend(max_conf.cpu().tolist())

                    img_feats  = trainer.model.imfeatures
                    txt_feats   = trainer.model.textfeatures
                    image_features_list.append(img_feats.cpu())
                    text_features_list.append(txt_feats.cpu())

            acc, ece = compute_metrics(result_dict)
            metrics_result[f"{style} | seed {seed}"] = {"acc": acc, "ece": ece}

            all_image_feats.append(torch.cat(image_features_list))
            all_text_feats.append(torch.cat(text_features_list))
            all_labels.extend([style] * text_features_list[-1].size(0))

    return torch.cat(all_image_feats), torch.cat(all_text_feats), all_labels, metrics_result


def save_metrics(metrics, output_dir):
    path = os.path.join(output_dir, "ece_accuracy_new_per_prompt_1.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("Prompt Style\tSeed\tAccuracy (%)\tECE (%)\n")
        for key, val in metrics.items():
            style, seed = key.split(" | seed ")
            f.write(f"{style}\t{seed}\t{val['acc']:.2f}\t{val['ece']:.2f}\n")
    print(f"[✓] Saved: {path}")


def main(args):
    prompt_styles = [
    "a clean photo of a",
    "a cropped image of a",
    "a high-resolution photo of a",
    "a poorly lit photo of a",
    "an overexposed image of a",
    "a realistic photo of a",
    "a synthetic image of a",
    "a diagram of a",
    "a shadowy photo of a",
    "a silhouette of a",

    "a vintage photo of a",
    "a dark image of a",
    "an infrared image of a",
    "a thermal image of a",
    "an x-ray of a",
    "a noisy image of a",
    "a watercolor painting of a",
    "a pencil sketch of a",
    "a computer-generated image of a",
    "an abstract painting of a",

    "a photo taken in daylight of a",
    "a nighttime image of a",
    "an aerial view of a",
    "a satellite image of a",
    "an underwater photo of a",
    "a zoomed-out photo of a",
    "a motion-blurred photo of a",
    "a photo from a surveillance camera of a",
    "a webcam image of a",
    "a drone shot of a",

    "a cinematic photo of a",
    "a dramatized image of a",
    "an editorial photo of a",
    "a fashion-style photo of a",
    "a wildlife photograph of a",
    "a scientific illustration of a",
    "an educational diagram of a",
    "a product photo of a",
    "a museum photo of a",
    "a passport-style photo of a"
]
    seeds = [1]

    print("[*] Running evaluation and feature extraction...")
    image_feats, text_feats, labels, metrics = evaluate_and_visualize(prompt_styles, seeds, args)

    print("[*] Running t-SNE...")
    tsne_img = TSNE(n_components=2, random_state=42).fit_transform(image_feats.numpy())
    tsne_txt = TSNE(n_components=2, random_state=42).fit_transform(text_feats.numpy())

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract dataset name from dataset config file path
    dataset_name = os.path.splitext(os.path.basename(args.dataset_config_file))[0]

    # Save t-SNE plots with dataset name
    text_tsne_path = os.path.join(args.output_dir, f"{dataset_name}_tsne-new_text_1.png")
    image_tsne_path = os.path.join(args.output_dir, f"{dataset_name}_tsne-new_image_1.png")

    plot_tsne(tsne_txt, labels, f"Prompt t-SNE (Text Features) - {dataset_name}", text_tsne_path)
    plot_tsne(tsne_img, labels, f"Prompt t-SNE (Image Features) - {dataset_name}", image_tsne_path)

    save_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Dataset root")
    parser.add_argument("--config-file", type=str, help="Trainer config YAML")
    parser.add_argument("--dataset-config-file", type=str, help="Dataset config YAML")
    parser.add_argument("--model-dir", type=str, help="Base dir with seed subfolders")
    parser.add_argument("--load-epoch", type=int, default=5, help="Epoch to load")
    parser.add_argument("--output-dir", type=str, default="./eval_tsne_outputs", help="Output path")
    parser.add_argument("--subsample-classes", type=str, default="all", help="Which class split to use: all, base, or new")
    args = parser.parse_args()
    main(args)