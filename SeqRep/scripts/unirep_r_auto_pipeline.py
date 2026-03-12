"""
run_unirep_pipeline.py

Auto pipeline:
1. Run seq2rep 3 times
2. Plot PCA + t-SNE for reproducibility
"""

import subprocess
import os

# ========= Config =========
FASTA = "../data/inputs/random_100.fasta"
WEIGHTS = "../unirep_restructured/64_weights"
OUTPUT_DIR = "../data/outputs"
PLOT_SCRIPT = "plot_tsne.py"
TITLE = "mLSTM (TF2.x)"
# ==========================


def run_seq2rep(run_id):
    output_file = f"{OUTPUT_DIR}/mlstm/runs/local_test{run_id}.h5"
    print(f"\n=== Running UniRep Run {run_id} ===")

    cmd = [
        "python", "../unirep_restructured/seq2rep.py",
        FASTA,
        WEIGHTS,
        output_file
    ]

    subprocess.run(cmd, check=True)
    return output_file


def run_plot(run1, run2, run3):
    print("\n=== Plotting PCA + t-SNE ===")

    output_plot = f"{OUTPUT_DIR}/mlstm/plot/tsne_mlstm_local1.png"

    cmd = [
        "python", PLOT_SCRIPT,
        "--run1", run1,
        "--run2", run2,
        "--run3", run3,
        "--output", output_plot,
        "--title", TITLE
    ]

    subprocess.run(cmd, check=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Run seq2rep 3 times
    h5_files = []
    for i in range(1, 4):
        h5_files.append(run_seq2rep(i))

    # Step 2: Plot
    run_plot(h5_files[0], h5_files[1], h5_files[2])

    print("\nDone! Pipeline finished successfully.")


if __name__ == "__main__":
    main()