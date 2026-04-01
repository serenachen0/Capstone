"""
Generate t-SNE plot to visualize reproducibility of embeddings.
Processes each run independently (PCA + t-SNE per run), then plots all on the same axes.
Matches the researcher's notebook approach.
"""

import os

# Reduce BLAS thread contention/crashes on macOS system Python + OpenBLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import time
import argparse


def load_embeddings(h5_path):
    """Load embeddings from an h5 file in sorted key order."""
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f.keys())
        Xs = np.array([f[k][:] for k in keys])
    return Xs


def process_run(Xs, label, pca_variance, perplexity, n_iter, seed):
    """Standardize -> PCA -> t-SNE for a single run's embeddings."""
    print(f"\nInput representation: {label}")
    print(f"Original feature shape = {Xs.shape}")

    Xs = np.asarray(Xs)
    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)

    n_samples, n_features = Xs.shape

    # Edge case: too few samples/features for PCA+tSNE
    if n_samples < 2 or n_features < 2:
        print("Insufficient dimensions for PCA/t-SNE, using fallback 2D projection.")
        out = np.zeros((n_samples, 2), dtype=float)
        if n_samples > 1:
            out[:, 0] = np.linspace(-1, 1, n_samples)
        return out

    # Standardize
    Xs = StandardScaler().fit_transform(Xs)

    # PCA (robust to low-rank inputs)
    try:
        pca = PCA(n_components=pca_variance, random_state=seed)
        Xs_pca = pca.fit_transform(Xs)
        print(f"Total number of selected principle components = {len(pca.explained_variance_ratio_)}")
        print(f"Total explained variance = {pca.explained_variance_ratio_.sum():.4f}")
    except ValueError as e:
        print(f"PCA fallback due to: {e}")
        k = min(n_samples, n_features)
        pca = PCA(n_components=k, random_state=seed, svd_solver="full")
        Xs_pca = pca.fit_transform(Xs)

    print(f"After PCA feature shape = {Xs_pca.shape}")

    # Ensure at least 2 features before t-SNE and avoid TSNE init='pca' failure on 1D inputs
    if Xs_pca.ndim == 1:
        Xs_pca = Xs_pca.reshape(-1, 1)
    if Xs_pca.shape[1] < 2:
        Xs_pca = np.hstack([Xs_pca, np.zeros((Xs_pca.shape[0], 1), dtype=Xs_pca.dtype)])

    # t-SNE
    perp = min(perplexity, len(Xs_pca) - 1)
    t0 = time()
    try:
        tsne = TSNE(
            n_components=2,
            verbose=1,
            perplexity=perp,
            max_iter=n_iter,
            random_state=seed,
            init="random",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            verbose=1,
            perplexity=perp,
            n_iter=n_iter,
            random_state=seed,
            init="random",
        )
    Xs_pca_tsne = tsne.fit_transform(Xs_pca)
    t1 = time()
    print(f"After PCA & t-SNE feature shape = {Xs_pca_tsne.shape}, perplexity = {perp}, finished in {t1 - t0:.2g} sec")

    return Xs_pca_tsne


def main():
    parser = argparse.ArgumentParser(description="Plot t-SNE of embeddings from multiple runs")
    parser.add_argument("--run1", type=str, required=True, help="First run h5 file")
    parser.add_argument("--run2", type=str, required=True, help="Second run h5 file")
    parser.add_argument("--run3", type=str, default=None, help="Third run h5 file (optional)")
    parser.add_argument("--output", type=str, default="tsne_plot.png", help="Output image file")
    parser.add_argument("--title", type=str, default="Embeddings", help="Plot title")
    parser.add_argument("--perplexity", type=float, default=50.0, help="t-SNE perplexity")
    parser.add_argument("--n_iter", type=int, default=10000, help="t-SNE max iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pca_variance", type=float, default=0.95, help="PCA explained variance threshold")
    args = parser.parse_args()

    runs = [("Run #1", args.run1, 'blue'),
            ("Run #2", args.run2, 'green')]
    if args.run3:
        runs.append(("Run #3", args.run3, 'orange'))

    # Plot setup — matches researcher's style
    mpl.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(args.title, y=0.95)

    for i, (label, h5_path, color) in enumerate(runs):
        Xs = load_embeddings(h5_path)
        Xs_2d = process_run(Xs, label, args.pca_variance, args.perplexity, args.n_iter, args.seed)

        s = 60 - 10 * i  # decreasing marker size per run, matches notebook
        ax.scatter(Xs_2d[:, 0], Xs_2d[:, 1], c=color, label=label,
                   marker='o', alpha=1, s=s)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
              markerscale=2, handletextpad=0.2, fontsize=18)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()