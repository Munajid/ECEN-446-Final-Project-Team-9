# plot_ber.py
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot BER vs SNR from a BER result file.")
    parser.add_argument("ber_file", help="Path to BER text file (columns: SNR, BP, [CNN iter1, ...])")
    parser.add_argument("--title", default="BER vs SNR", help="Plot title")
    parser.add_argument("--out", default=None, help="Output image filename (e.g., ber_plot.png)")
    args = parser.parse_args()

    # Load data
    data = np.loadtxt(args.ber_file)
    if data.ndim == 1:
        data = data[None, :]  # handle single-row edge case

    snr_db = data[:, 0]
    ber_bp = data[:, 1]
    ber_cnn = data[:, -1]  # last column = final BP-CNN iteration (works even if only 2 cols)

    print("Loaded file:", args.ber_file)
    print("SNR points:", snr_db)
    print("BP BER:", ber_bp)
    print("BP-CNN BER (final):", ber_cnn)

    # Plot
    plt.figure()
    plt.semilogy(snr_db, ber_bp, "o-", label="BP only")
    plt.semilogy(snr_db, ber_cnn, "s-", label="BP-CNN (final iter)")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Output
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.ber_file))[0]
        out_name = base + "_ber.png"
    else:
        out_name = args.out

    plt.savefig(out_name, dpi=300)
    print("Saved figure to:", out_name)
    plt.show()


if __name__ == "__main__":
    main()