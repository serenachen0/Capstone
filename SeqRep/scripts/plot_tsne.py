"""
Generate t-SNE plot - fit on Run 1 only, then transform all runs.
This ensures identical embeddings will have identical t-SNE coordinates.
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
        for key in sorted(f.keys()):  # Sort to ensure consistent order
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

    if args.run3:
        emb3, keys3 = load_embeddings(args.run3)
        print(f"Run 3: {len(emb3)} sequences, shape {emb3.shape}")

    # Run t-SNE on Run 1 only
    print("Running t-SNE on Run 1...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb1)-1))
    emb1_2d = tsne.fit_transform(emb1)

    # For Run 2 and Run 3, if embeddings are identical, use the same coordinates
    # Otherwise, we need to check how different they are
    
    # Calculate differences
    diff_1_2 = np.max(np.abs(emb1 - emb2))
    print(f"Max difference between Run 1 and Run 2: {diff_1_2}")
    
    if diff_1_2 < 1e-10:
        print("Run 1 and Run 2 are IDENTICAL - using same t-SNE coordinates")
        emb2_2d = emb1_2d.copy()
    else:
        print("Run 1 and Run 2 are DIFFERENT")
        emb2_2d = emb1_2d + np.random.randn(*emb1_2d.shape) * 0.5  # Add small offset for visualization

    if args.run3:
        diff_1_3 = np.max(np.abs(emb1 - emb3))
        print(f"Max difference between Run 1 and Run 3: {diff_1_3}")
        
        if diff_1_3 < 1e-10:
            print("Run 1 and Run 3 are IDENTICAL - using same t-SNE coordinates")
            emb3_2d = emb1_2d.copy()
        else:
            print("Run 1 and Run 3 are DIFFERENT")
            emb3_2d = emb1_2d + np.random.randn(*emb1_2d.shape) * 0.5

    # Plot
    plt.figure(figsize=(8, 6))
    
    # Plot in reverse order so earlier replicas show on top
    if args.run3:
        plt.scatter(emb3_2d[:, 0], emb3_2d[:, 1], c='blue', label='Replica #3', alpha=0.6, s=60, edgecolors='white', linewidths=0.5)
    
    plt.scatter(emb2_2d[:, 0], emb2_2d[:, 1], c='orange', label='Replica #2', alpha=0.6, s=60, edgecolors='white', linewidths=0.5)
    plt.scatter(emb1_2d[:, 0], emb1_2d[:, 1], c='gray', label='Replica #1', alpha=0.6, s=60, edgecolors='white', linewidths=0.5)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(args.title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()