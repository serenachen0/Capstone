"""
Auto pipeline: run ESM extract 3 times -> convert to h5 -> compare -> plot t-SNE
"""

import os
import sys
import subprocess
import argparse
import torch
import h5py
import numpy as np

# ========== Config ==========
DEFAULT_FASTA = "../data/inputs/random_100.fasta"
OUTPUT_DIR = "../data/outputs/esm/runs"
PLOT_DIR = "../data/outputs/esm/plot"
MODEL = "esm2_t33_650M_UR50D"
LAYER = 33
TSNE_SCRIPT = "../scripts/plot_tsne.py"
# ==========================

def run_esm_extract(run_id, fasta_path):
    out_dir = f"{OUTPUT_DIR}/esm_run{run_id}"
    print(f"\n=== Running ESM extract Run {run_id} ===")
    print(f"Using FASTA: {fasta_path}")
    cmd = [
        sys.executable, "../esm/scripts/extract.py",
        MODEL, fasta_path, out_dir,
        "--repr_layers", str(LAYER),
        "--include", "mean"
    ]
    subprocess.run(cmd, check=True)
    return out_dir

def pt_to_h5(pt_dir, h5_path, layer=33):
    print(f"Converting {pt_dir} -> {h5_path}")
    with h5py.File(h5_path, 'w') as h5f:
        for pt_file in sorted(os.listdir(pt_dir)):
            if pt_file.endswith('.pt'):
                data = torch.load(
                    os.path.join(pt_dir, pt_file),
                    weights_only=False
                )
                key = pt_file.replace('.pt', '')
                embedding = data['mean_representations'][layer].numpy()
                h5f.create_dataset(key, data=embedding)
    print(f"Saved {h5_path}")

def compare_h5(h5_path1, h5_path2, label):
    with h5py.File(h5_path1, 'r') as f1, h5py.File(h5_path2, 'r') as f2:
        emb1 = np.array([f1[k][:] for k in sorted(f1.keys())])
        emb2 = np.array([f2[k][:] for k in sorted(f2.keys())])
    diff = np.max(np.abs(emb1 - emb2))
    identical = np.allclose(emb1, emb2)
    print(f"{label}: max diff = {diff}, identical = {identical}")

def plot_tsne(h5_run1, h5_run2, h5_run3, output, title):
    print(f"\n=== Plotting t-SNE: {title} ===")
    cmd = [
        sys.executable, TSNE_SCRIPT,
        "--run1", h5_run1,
        "--run2", h5_run2,
        "--run3", h5_run3,
        "--output", output,
        "--title", title
    ]

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

def resolve_fasta(cli_fasta=None):
    return cli_fasta or os.getenv("FASTA_FILE") or os.getenv("FASTA") or DEFAULT_FASTA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, default=None)
    args = parser.parse_args()

    fasta_path = resolve_fasta(args.fasta)
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Step 1: Run ESM 3 times
    for i in range(1, 4):
        run_esm_extract(i, fasta_path)

    # Step 2: Convert pt -> h5
    h5_files = []
    for i in range(1, 4):
        pt_dir = f"{OUTPUT_DIR}/esm_run{i}"
        h5_path = f"{OUTPUT_DIR}/esm_run{i}.h5"
        pt_to_h5(pt_dir, h5_path, layer=LAYER)
        h5_files.append(h5_path)

    # Step 3: Compare embeddings
    print("\n=== Comparing embeddings ===")
    compare_h5(h5_files[0], h5_files[1], "Run1 vs Run2")
    compare_h5(h5_files[0], h5_files[2], "Run1 vs Run3")

    # Step 4: Plot t-SNE
    plot_tsne(
        h5_files[0], h5_files[1], h5_files[2],
        f"{PLOT_DIR}/tsne_esm.png",
        "ESM"
    )
    print(f"\nDone! Plot saved to {PLOT_DIR}/tsne_esm.png")

if __name__ == "__main__":
    main()