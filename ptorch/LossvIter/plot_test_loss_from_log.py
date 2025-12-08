#!/usr/bin/env python3
"""
Parse raw training log and plot test loss vs epoch.

Expected lines in the log (examples):

    Test loss: 0.307154
    Start training...
    Epoch 500, train loss: 0.296638
    Test loss: 0.294405
    Epoch 1000, train loss: 0.288351
    Test loss: 0.294145
    ...

Logic:
- The first "Test loss: X" seen *before* any "Epoch N" is treated as epoch 0.
- After that, each "Test loss: X" is associated with the most recent "Epoch N"
  seen above it.
"""

import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    epochs = []
    test_losses = []

    current_epoch_for_next_test = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # Match lines like: "Epoch 500, train loss: 0.296638"
            m_epoch = re.search(r"Epoch\s+(\d+)", line)
            if m_epoch:
                current_epoch_for_next_test = int(m_epoch.group(1))
                continue

            # Match lines like: "Test loss: 0.307154"
            m_test = re.search(r"Test loss:\s*([0-9.eE+-]+)", line)
            if m_test:
                loss = float(m_test.group(1))

                # If we haven't seen an Epoch yet, treat as epoch 0
                if current_epoch_for_next_test is None:
                    epoch = 0
                else:
                    epoch = current_epoch_for_next_test

                epochs.append(epoch)
                test_losses.append(loss)

                # We consumed this epoch marker for this test loss
                # (optional; keeps one test per epoch)
                current_epoch_for_next_test = None

    if len(epochs) == 0:
        raise RuntimeError(f"No 'Test loss:' lines found in {log_path}")

    return np.array(epochs, dtype=float), np.array(test_losses, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Parse training log and plot test loss vs epoch."
    )
    parser.add_argument("log_file", help="Path to raw training log text file")
    parser.add_argument(
        "--title",
        default="Test loss vs epoch",
        help="Plot title",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image filename (e.g., test_loss_plot.png). "
             "If omitted, uses <log_file_basename>_loss.png.",
    )

    args = parser.parse_args()

    epochs, test_losses = parse_log_file(args.log_file)

    print("Parsed epochs:", epochs)
    print("Parsed test losses:", test_losses)

    plt.figure()
    plt.plot(epochs, test_losses, "o-", label="Test loss")

    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.log_file))[0]
        out_name = base + "_loss.png"
    else:
        out_name = args.out

    plt.savefig(out_name, dpi=300)
    print(f"Saved figure to: {out_name}")
    plt.show()


if __name__ == "__main__":
    main()
