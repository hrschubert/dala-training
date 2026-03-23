#!/usr/bin/env python3
"""Plot training loss and learning rate from lczero training logs."""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    pattern = re.compile(
        r"Step (\d+) .*, Loss: ([\d.]+), LR: ([\d.eE+-]+), "
        r"policy/vanilla: ([\d.]+), value/winner: ([\d.]+)"
    )
    steps, total_loss, lrs, policy, value = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                total_loss.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                policy.append(float(m.group(4)))
                value.append(float(m.group(5)))
    return (np.array(steps), np.array(total_loss), np.array(lrs),
            np.array(policy), np.array(value))

def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "train_full_d.log"
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

    # LR on right axis
    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color="tab:orange", alpha=0.6, linewidth=1.2,
             linestyle="--", label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=8)

    ax1.set_title("Dala 1100 Training")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
