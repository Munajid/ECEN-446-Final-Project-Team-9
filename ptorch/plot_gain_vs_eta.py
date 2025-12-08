#!/usr/bin/env python3
"""
Plot performance gain (dB) vs eta from a text file with two columns:
    eta   gain_dB
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot performance gain vs η.")
    parser.add_argument("infile", help="Input text file (eta, gain_dB).")
    parser.add_argument("--out", default="gain_vs_eta.png",
                        help="Output image filename (default: gain_vs_eta.png).")
    parser.add_argument("--title", default="Performance Gain vs η",
                        help="Plot title.")
    args = parser.parse_args()

    data = np.loadtxt(args.infile)
    if data.ndim == 1:
        data = data[None, :]

    eta = data[:, 0]
    gain = data[:, 1]

    plt.figure()
    plt.plot(eta, gain, marker="o", linestyle="-")
    plt.xlabel("η")
    plt.ylabel("Performance Gains (dB)")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
