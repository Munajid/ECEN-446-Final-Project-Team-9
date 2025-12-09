#!/usr/bin/env python3
"""
Plot residual noise metric vs SNR from a text file.

Typical input (for update_llr_with_epdf = False) looks like:

    SNR0   metric0
    SNR1   metric1
    ...

Usage examples:

    # Simple default use
    python plot_residual_vs_snr.py model/residual_noise_property_netid0_model0.txt

    # With custom labels / filename
    python plot_residual_vs_snr.py model/residual_noise_property_netid0_model0.txt \
        --label "Residual noise power" \
        --title "Residual noise vs SNR (η=0.8)" \
        --out residual_vs_snr_eta08.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot residual noise metric vs SNR from a text file."
    )
    parser.add_argument(
        "infile",
        help="Input text file (e.g., residual_noise_property_netid0_model0.txt)",
    )
    parser.add_argument(
        "--snr-col",
        type=int,
        default=0,
        help="0-based column index for SNR (default: 0).",
    )
    parser.add_argument(
        "--val-col",
        type=int,
        default=1,
        help="0-based column index for residual noise metric (default: 1).",
    )
    parser.add_argument(
        "--label",
        default="Residual noise",
        help="Y-axis label / legend label (default: 'Residual noise').",
    )
    parser.add_argument(
        "--out",
        default="residual_vs_snr.png",
        help="Output image filename (default: residual_vs_snr.png).",
    )
    parser.add_argument(
        "--title",
        default="Residual noise vs SNR",
        help="Plot title (default: 'Residual noise vs SNR').",
    )

    args = parser.parse_args()

    # Load numeric data (assumes whitespace-delimited)
    data = np.loadtxt(args.infile)

    # Handle single-line edge case
    if data.ndim == 1:
        data = data[None, :]

    if args.snr_col >= data.shape[1] or args.val_col >= data.shape[1]:
        raise ValueError(
            f"File {args.infile} has {data.shape[1]} columns, "
            f"but snr-col={args.snr_col}, val-col={args.val_col}"
        )

    snr = data[:, args.snr_col]
    val = data[:, args.val_col]

    plt.figure()
    # Residual noise is usually plotted on a log scale
    plt.semilogy(snr, val, marker="o", linestyle="-")
    plt.xlabel("SNR (dB)")
    plt.ylabel(args.label)
    plt.title(args.title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
