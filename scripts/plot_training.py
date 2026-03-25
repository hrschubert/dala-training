#!/usr/bin/env python3
"""Plot training loss and learning rate from lczero training logs."""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    # Extract fields individually for robustness across log format changes
    re_step = re.compile(r"Step (\d+)\b")
    re_loss = re.compile(r"Loss:\s*([\d.eE+-]+)")
    re_lr = re.compile(r"LR:\s*([\d.eE+-]+)")
    re_policy = re.compile(r"'?policy/vanilla'?:\s*([\d.eE+-]+)")
    re_value = re.compile(r"'?value/winner'?:\s*([\d.eE+-]+)")

    steps, total_loss, lrs, policy, value = [], [], [], [], []
    with open(path) as f:
        for line in f:
            ms = re_step.search(line)
            ml = re_loss.search(line)
            mp = re_policy.search(line)
            mv = re_value.search(line)
            if not (ms and ml and mp and mv):
                continue
            steps.append(int(ms.group(1)))
            total_loss.append(float(ml.group(1)))
            policy.append(float(mp.group(1)))
            value.append(float(mv.group(1)))
            mlr = re_lr.search(line)
            lrs.append(float(mlr.group(1)) if mlr else float('nan'))
    return (np.array(steps), np.array(total_loss), np.array(lrs),
            np.array(policy), np.array(value))

def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")

def main():
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "training_plot.pdf"

    steps, total_loss, lrs, policy, value = parse_log(log_path)
    print(f"Parsed {len(steps)} steps (range {steps[0]}–{steps[-1]})")

    w = 50
    steps_s = steps[w-1:]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Policy loss on left axis
    ax1.plot(steps_s, smooth(policy, w), color="tab:blue", alpha=0.9,
             linewidth=0.8, label="policy/vanilla (smoothed)")
    ax1.plot(steps_s, smooth(value, w), color="tab:green", alpha=0.9,
             linewidth=0.8, label="value/winner (smoothed)")
    ax1.plot(steps_s, smooth(total_loss, w), color="tab:red", alpha=0.7,
             linewidth=0.8, label="total loss (smoothed)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(bottom=0)

    # LR on right axis (only if LR data is available)
    has_lr = not np.all(np.isnan(lrs))
    if has_lr:
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, color="tab:orange", alpha=0.6, linewidth=1.2,
                 linestyle="--", label="Learning Rate")
        ax2.set_ylabel("Learning Rate", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_lr:
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=8)

    ax1.set_title("Dala Training")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
