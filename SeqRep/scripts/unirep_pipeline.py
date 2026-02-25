"""
Auto pipeline: run UniRep seq2rep 3 times -> compare -> plot t-SNE

Script location : Capstone/SeqRep/scripts/unirep_pipeline.py
Working directory: Capstone/SeqRep/scripts/
"""

import os
import subprocess
import sys

# ──────────────────────────────────────────────
# Anchor every path to THIS file so the script
# works regardless of where Python is invoked.
# ──────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))   # Capstone/SeqRep/scripts
SEQREP_DIR   = os.path.dirname(THIS_DIR)                    # Capstone/SeqRep
CAPSTONE_DIR = os.path.dirname(SEQREP_DIR)                  # Capstone
UNIREP_DIR   = os.path.join(SEQREP_DIR, "unirep")           # Capstone/SeqRep/unirep

# Use the same python executable that is running this script
# so subprocess inherits the virtualenv with TensorFlow installed.
PYTHON = sys.executable

# ========== Config ==========
MODEL_SIZE = "1900"

FASTA_FILE     = os.path.join(SEQREP_DIR,  "data", "inputs",  "random_100.fasta")
OUTPUT_DIR     = os.path.join(SEQREP_DIR,  "data", "outputs")
PLOT_DIR       = os.path.join(OUTPUT_DIR,  "plot")
MODEL_PATH     = os.path.join(SEQREP_DIR,  "data", "weights", f"unirep{MODEL_SIZE}")

SEQ2REP_SCRIPT = os.path.join(UNIREP_DIR,  "scripts", "seq2rep.py")
COMPARE_SCRIPT = os.path.join(UNIREP_DIR,  "scripts", "reps-compare.py")
TSNE_SCRIPT    = os.path.join(THIS_DIR,    "plot_tsne.py")
# ============================


def run_seq2rep(run_id):
    """Run seq2rep.py for one run and return the absolute output .h5 path."""
    out_abs        = os.path.join(OUTPUT_DIR, f"unirep_run{run_id}.h5")
    script_dir     = os.path.dirname(SEQ2REP_SCRIPT)   # Capstone/unirep/scripts

    print(f"\n=== Running UniRep seq2rep Run {run_id} ===")

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Inject both unirep/scripts AND Capstone onto sys.path so all
    # relative imports inside seq2rep.py resolve correctly.
    cmd = [
        PYTHON, "-c",
        (
            f"import sys; "
            f"sys.path.insert(0, '{script_dir}'); "
            f"sys.path.insert(0, '{CAPSTONE_DIR}'); "
            "import tensorflow as tf; "
            "tf.set_random_seed = tf.compat.v1.set_random_seed; "
            "tf.compat.v1.disable_eager_execution(); "
            f"import runpy; runpy.run_path('{SEQ2REP_SCRIPT}', run_name='__main__')"
        ),
        FASTA_FILE, MODEL_PATH, out_abs,
    ]

    subprocess.run(cmd, check=True, env=env)
    return out_abs   # always absolute


def compare_h5(h5_path1, h5_path2, label):
    """Call reps-compare.py to compare two .h5 files."""
    print(f"\n--- {label} ---")
    cmd = [PYTHON, COMPARE_SCRIPT, h5_path1, h5_path2]
    subprocess.run(cmd, check=True)


def plot_tsne(h5_run1, h5_run2, h5_run3, output, title):
    """Call plot_tsne.py as a subprocess."""
    print(f"\n=== Plotting t-SNE: {title} ===")
    cmd = [
        PYTHON, TSNE_SCRIPT,
        "--run1",   h5_run1,
        "--run2",   h5_run2,
        "--run3",   h5_run3,
        "--output", output,
        "--title",  title,
    ]
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    # Step 1: Run UniRep seq2rep 3 times
    h5_files = [run_seq2rep(i) for i in range(1, 4)]

    # Step 2: Compare embeddings
    print("\n=== Comparing embeddings ===")
    compare_h5(h5_files[0], h5_files[1], "Run1 vs Run2")
    compare_h5(h5_files[0], h5_files[2], "Run1 vs Run3")

    # Step 3: Plot t-SNE
    tsne_out = os.path.join(PLOT_DIR, "tsne_unirep.png")
    plot_tsne(h5_files[0], h5_files[1], h5_files[2], tsne_out, "UniRep")

    print(f"\nDone! Plot saved to {tsne_out}")


if __name__ == "__main__":
    main()