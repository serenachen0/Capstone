"""
Generate t-SNE plot to visualize reproducibility of mLSTM embeddings.
Compares multiple runs to check if results are deterministic.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

def load_embeddings(h5_path):
    """Load all embeddings from an h5 file."""
    embeddings = []
    keys = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            embeddings.append(f[key][:])
            keys.append(key)
    return np.array(embeddings), keys

def main():
    parser = argparse.ArgumentParser(description="Plot t-SNE of embeddings from multiple runs")
    parser.add_argument("--run1", type=str, required=True, help="First run h5 file")
    parser.add_argument("--run2", type=str, required=True, help="Second run h5 file")
    parser.add_argument("--run3", type=str, default=None, help="Third run h5 file (optional)")
    parser.add_argument("--output", type=str, default="tsne_plot.png", help="Output image file")
    parser.add_argument("--title", type=str, default="mLSTM", help="Plot title")
    args = parser.parse_args()

    # Load embeddings from each run
    emb1, keys1 = load_embeddings(args.run1)
    emb2, keys2 = load_embeddings(args.run2)
    
    print(f"Run 1: {len(emb1)} sequences, shape {emb1.shape}")
    print(f"Run 2: {len(emb2)} sequences, shape {emb2.shape}")

    # Combine embeddings
    if args.run3:
        emb3, keys3 = load_embeddings(args.run3)
        print(f"Run 3: {len(emb3)} sequences, shape {emb3.shape}")
        all_embeddings = np.vstack([emb1, emb2, emb3])
        labels = ['Replica #1'] * len(emb1) + ['Replica #2'] * len(emb2) + ['Replica #3'] * len(emb3)
    else:
        all_embeddings = np.vstack([emb1, emb2])
        labels = ['Replica #1'] * len(emb1) + ['Replica #2'] * len(emb2)

    print(f"Total embeddings: {len(all_embeddings)}")

    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    colors = {'Replica #1': 'gray', 'Replica #2': 'orange', 'Replica #3': 'blue'}
    
    for label in ['Replica #1', 'Replica #2', 'Replica #3']:
        mask = [l == label for l in labels]
        if sum(mask) > 0:
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[label],
                label=label,
                alpha=0.7,
                s=50
            )

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(args.title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()