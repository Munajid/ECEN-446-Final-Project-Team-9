#!/usr/bin/env python3
import argparse
import numpy as np
import os

def find_snr_at_target(snr_db, ber, target_ber):
    """
    Given arrays snr_db[i], ber[i] (monotonically decreasing BER vs SNR),
    linearly interpolate (in log10 BER space) to find SNR where BER = target_ber.
    """
    log_target = np.log10(target_ber)
    log_ber = np.log10(ber)

    for i in range(len(snr_db) - 1):
        y1, y2 = log_ber[i], log_ber[i + 1]
        x1, x2 = snr_db[i], snr_db[i + 1]

        # Look for interval where BER crosses the target (y1 > target >= y2)
        if y1 > log_target >= y2:
            m = (y2 - y1) / (x2 - x1)
            x_target = x1 + (log_target - y1) / m
            return x_target

    raise RuntimeError(
        f"Target BER {target_ber} not bracketed by provided points."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Compute SNR gain (BP vs BP-CNN) at a target BER for one η."
    )
    parser.add_argument("ber_file", help="BER results file (columns: SNR, BP, CNN)")
    parser.add_argument("--eta", type=float, required=True, help="Channel correlation η")
    parser.add_argument("--target-ber", type=float, default=1e-3,
                        help="Target BER for defining gain (default: 1e-3)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output text file to append 'eta  gain_dB' to")
    args = parser.parse_args()

    data = np.loadtxt(args.ber_file)
    if data.ndim == 1:
        data = data[None, :]

    snr_db = data[:, 0]
    ber_bp = data[:, 1]
    ber_cnn = data[:, -1]

    snr_bp = find_snr_at_target(snr_db, ber_bp, args.target_ber)
    snr_cnn = find_snr_at_target(snr_db, ber_cnn, args.target_ber)
    gain = snr_bp - snr_cnn

    print(f"η = {args.eta:.2f}")
    print(f"SNR_BP    @ BER={args.target_ber:g}: {snr_bp:.3f} dB")
    print(f"SNR_BPCNN @ BER={args.target_ber:g}: {snr_cnn:.3f} dB")
    print(f"Gain = {gain:.3f} dB")

    if args.out is not None:
        mode = "a" if os.path.exists(args.out) else "w"
        with open(args.out, mode) as f:
            f.write(f"{args.eta:.2f}\t{gain:.6f}\n")
        print(f"Appended to {args.out}")

if __name__ == "__main__":
    main()
