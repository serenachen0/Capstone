"""
plot_tsne.py - Plot t-SNE from ESM embedding runs
Per-run StandardScaler -> PCA (95%) -> t-SNE -> overlay on same axes
"""
import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from time import time


# ========== Args ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Plot t-SNE of embeddings from multiple runs")
    parser.add_argument("--run1",   type=str, required=True,  help="First run h5 file")
    parser.add_argument("--run2",   type=str, required=True,  help="Second run h5 file")
    parser.add_argument("--run3",   type=str, default=None,   help="Third run h5 file (optional)")
    parser.add_argument("--output", type=str, default="tsne_plot.png", help="Output image file")
    parser.add_argument("--title",  type=str, default="t-SNE", help="Plot title")
    return parser.parse_args()


# ========== Load ==========
def load_h5(h5_path):
    """Load all embeddings from h5 file, sorted by key."""
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f.keys())
        embeddings = np.array([f[k][:] for k in keys])
    return embeddings, keys


# ========== PCA + t-SNE ==========
def pca_then_tsne(Xs, label, perplexity=50, n_iter=10000, random_state=42):
    """Standardize -> PCA (95% variance) -> t-SNE. Matches notebook logic."""

    print(f"Input representation: {label}")
    print(f"Original feature shape = {Xs.shape}")

    # standardize
    Xs = StandardScaler().fit_transform(Xs)

    # PCA
    pca = PCA(n_components=0.95)
    Xs_pca = pca.fit_transform(Xs)
    total_pcs = len(pca.explained_variance_ratio_)
    total_var = pca.explained_variance_ratio_.sum()
    print(f"Total number of selected principle components = {total_pcs}")
    print(f"Total explained variance = {total_var}")
    print(f"After PCA feature shape = {Xs_pca.shape}")

    # t-SNE
    t0 = time()
    Xs_tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state
    ).fit_transform(Xs_pca)
    t1 = time()
    print(f"After PCA & t-SNE feature shape = {Xs_tsne.shape}, "
          f"perplexity = {perplexity} finished in {t1 - t0:.2g} sec")

    return Xs_tsne


# ========== Plot ==========
def plot(runs_tsne, labels, colors, title, output_path):
    mpl.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title, y=0.95)

    for i, (Xs_tsne, label, color) in enumerate(zip(runs_tsne, labels, colors)):
        s = 60 - 10 * i   # decreasing point size: 60, 50, 40 (matches notebook)
        ax.scatter(
            Xs_tsne[:, 0],
            Xs_tsne[:, 1],
            c=color,
            label=label,
            marker='o',
            alpha=1,
            s=s
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
        markerscale=2,
        handletextpad=0.2,
        fontsize=18
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.clf()
    print(f"Saved plot to {output_path}")


# ========== Main ==========
def main():
    args = parse_args()

    h5_paths = [args.run1, args.run2]
    labels   = ["Run #1", "Run #2"]
    colors   = ["blue",   "green"]

    if args.run3:
        h5_paths.append(args.run3)
        labels.append("Run #3")
        colors.append("orange")

    PERPLEXITY   = 50
    N_ITER       = 10000
    RANDOM_STATE = 42

    runs_tsne = []
    for h5_path, label in zip(h5_paths, labels):
        emb, keys = load_h5(h5_path)
        Xs_tsne = pca_then_tsne(
            emb,
            label=h5_path,
            perplexity=PERPLEXITY,
            n_iter=N_ITER,
            random_state=RANDOM_STATE
        )
        runs_tsne.append(Xs_tsne)

    plot(runs_tsne, labels, colors, args.title, args.output)


if __name__ == "__main__":
    main()