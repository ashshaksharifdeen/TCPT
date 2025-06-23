"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std (or 95% CI) across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run:

    python tools/parse_test_res.py output/my_experiment

Add --ci95 for 95% confidence interval instead of std:

    python tools/parse_test_res.py output/my_experiment --ci95

For multiple experiments (each subfolder is its own experiment), add --multi-exp:

    python tools/parse_test_res.py output/my_experiment --multi-exp
"""

import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    """95% confidence interval half-width."""
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)
    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath), f"Missing log.txt in {subdir}"
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()

                # only start capturing after seeing the correct marker
                if args.test_log:
                    # for evaluation‐only logs, we begin at the "Total samples" line
                    if raw_line.startswith("=> Total samples:"):
                        good_to_go = True
                        continue
                else:
                    # for train+test logs, we still look for the training finish marker
                    if line == end_signal:
                        good_to_go = True
                        continue

                if not good_to_go:
                    continue

                # try each metric regex
                for metric in metrics:
                    m = metric["regex"].search(raw_line)
                    if m:
                        if "file" not in output:
                            output["file"] = fpath
                        output[metric["name"]] = float(m.group(1))

        if output:
            outputs.append(output)

    assert outputs, f"No metrics found in {directory}"

    # print per-seed results and collect values
    metrics_results = defaultdict(list)
    for out in outputs:
        msg = ""
        for k, v in out.items():
            if k == "file":
                msg += f"{v}  "
            else:
                msg += f"{k}: {v:.2f}%. "
                metrics_results[k].append(v)
        print(msg)

    # summary
    print("===")
    print(f"Summary of directory: {directory}")
    for k, vals in metrics_results.items():
        avg = np.mean(vals)
        std = compute_ci95(vals) if args.ci95 else np.std(vals)
        print(f"* {k}: {avg:.2f}% ± {std:.2f}%")
    print("===")

    return {k: np.mean(v) for k, v in metrics_results.items()}


def main(args, end_signal):
    # Define all metrics to extract:
    metrics = [
        {"name": "accuracy",      "regex": re.compile(r"Accuracy:\s*([\d\.eE+-]+)%", re.IGNORECASE)},
        {"name": "ece",           "regex": re.compile(r"ECE:\s*([\d\.eE+-]+)%",      re.IGNORECASE)},
        {"name": "mce",           "regex": re.compile(r"MCE:\s*([\d\.eE+-]+)%",      re.IGNORECASE)},
        {"name": "adaptive_ece",  "regex": re.compile(r"Adaptive ECE:\s*([\d\.eE+-]+)%", re.IGNORECASE)},
        {"name": "macro_f1",      "regex": re.compile(r"Macro-F1:\s*([\d\.eE+-]+)%",    re.IGNORECASE)},
        {"name": "piece",         "regex": re.compile(r"Piece:\s*([\d\.eE+-]+)%",      re.IGNORECASE)},
    ]

    if args.multi_exp:
        # average across multiple experiments
        final = defaultdict(list)
        for exp in listdir_nohidden(args.directory, sort=True):
            exp_dir = osp.join(args.directory, exp)
            res = parse_function(*metrics, directory=exp_dir, args=args, end_signal=end_signal)
            for k, v in res.items():
                final[k].append(v)

        print("Average performance across experiments")
        for k, vals in final.items():
            avg = np.mean(vals)
            std = compute_ci95(vals) if args.ci95 else np.std(vals)
            print(f"* {k}: {avg:.2f}% ± {std:.2f}%")
    else:
        # single experiment (directory = seeds folder)
        parse_function(*metrics, directory=args.directory, args=args, end_signal=end_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to experiment folder")
    parser.add_argument("--ci95",      action="store_true", help="use 95% CI instead of std")
    parser.add_argument("--test-log",  action="store_true", help="parse evaluation-only logs (=> Total samples)")
    parser.add_argument("--multi-exp", action="store_true", help="treat each subfolder as separate experiment")
    args = parser.parse_args()

    # for train+test runs we still look for "Finish training"
    end_signal = "Finish training"
    main(args, end_signal)
