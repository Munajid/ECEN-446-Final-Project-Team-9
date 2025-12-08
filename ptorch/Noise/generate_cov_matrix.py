#!/usr/bin/env python
"""
Generate the transfer matrix for colored noise, matching Generate_cov_1_2.m.

It builds a covariance matrix:
    cov[i,j] = eta ** abs(i-j)
of size N x N, then computes its matrix square root, and writes it
to a .dat file as single-precision floats (like MATLAB's 'single').

Important for this PyTorch/TF pipeline:
- We write the matrix in ROW-MAJOR order.
- DataIO.NoiseIO reads the file with np.fromfile(...).reshape(N, N),
  which assumes row-major layout, so this keeps everything consistent.
"""

import argparse
import numpy as np


def build_cov_matrix(eta: float, N: int) -> np.ndarray:
    """
    Build covariance matrix with entries cov[i,j] = eta ** |i-j|.
    Equivalent to the MATLAB double loop, but vectorized.
    """
    idx = np.arange(N)
    # shape (N, N), element-wise |i - j|
    diff = np.abs(idx[:, None] - idx[None, :])
    cov = eta ** diff
    return cov.astype(np.float64)


def matrix_sqrt_symmetric(A: np.ndarray) -> np.ndarray:
    """
    Matrix square root for a symmetric positive-definite matrix A
    using eigen-decomposition: A = Q Λ Q^T  ->  A^(1/2) = Q Λ^(1/2) Q^T.
    """
    # eigh is for symmetric/hermitian matrices
    eigvals, eigvecs = np.linalg.eigh(A)

    # Numerical safety: clip tiny negative eigenvalues to zero
    eigvals = np.clip(eigvals, 0.0, None)

    sqrt_vals = np.sqrt(eigvals)

    # Q diag(sqrt_vals) Q^T
    # (eigvecs * sqrt_vals) multiplies each column j of Q by sqrt_vals[j]
    tmp = eigvecs * sqrt_vals
    transfer = tmp @ eigvecs.T
    return transfer


def main():
    parser = argparse.ArgumentParser(
        description="Generate cov_1_2_corr_paraXX.dat transfer matrix (Python version)."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.8,
        help="Correlation parameter η (same as 'eta' in the MATLAB script).",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=576,
        help="Matrix dimension N (default 576).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .dat filename. If omitted, uses cov_1_2_corr_para%.2f.dat",
    )

    args = parser.parse_args()

    eta = args.eta
    N = args.N

    if args.out is None:
        out_name = f"cov_1_2_corr_para{eta:.2f}.dat"
    else:
        out_name = args.out

    print(f"Generating covariance for eta = {eta}, N = {N} ...")
    cov = build_cov_matrix(eta, N)

    print("Computing matrix square root (this can take a little while for N=576)...")
    transfer_mat = matrix_sqrt_symmetric(cov)

    # Write as float32, ROW-MAJOR.
    # DataIO.NoiseIO does: np.fromfile(...).reshape(N, N) (row-major),
    # so this keeps the matrix consistent with what TF expects.
    transfer_mat_f32 = transfer_mat.astype(np.float32)
    with open(out_name, "wb") as f:
        transfer_mat_f32.tofile(f)

    print(f"Done. Wrote transfer matrix to: {out_name}")
    print(f"Shape: {transfer_mat_f32.shape}, dtype: {transfer_mat_f32.dtype}")


if __name__ == "__main__":
    main()
