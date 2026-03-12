"""
Generate t-SNE plot to visualize reproducibility of embeddings.
Processes each run independently (PCA + t-SNE per run), then plots all on the same axes.
Matches the researcher's notebook approach.
"""

import os
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

    # Standardize
    Xs = StandardScaler().fit_transform(Xs)

    # PCA
    pca = PCA(n_components=pca_variance, random_state=seed)
    Xs_pca = pca.fit_transform(Xs)
    print(f"Total number of selected principle components = {len(pca.explained_variance_ratio_)}")
    print(f"Total explained variance = {pca.explained_variance_ratio_.sum():.4f}")
    print(f"After PCA feature shape = {Xs_pca.shape}")

    # t-SNE
    perp = min(perplexity, len(Xs_pca) - 1)
    t0 = time()
    Xs_pca_tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perp,
        max_iter=n_iter,
        random_state=seed,
    ).fit_transform(Xs_pca)
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