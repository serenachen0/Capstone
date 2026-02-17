"""
Generate randomized protein sequences for reproducibility testing.
Takes the original 5y0m sequence and shuffles the amino acids to create new sequences.
"""

import random
import argparse

def read_fasta(filepath):
    """Read a fasta file and return the sequence."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return sequence

def generate_random_sequences(original_seq, n=100, seed=42):
    """Generate n randomized sequences by shuffling the original sequence."""
    random.seed(seed)
    sequences = []
    for i in range(n):
        shuffled = list(original_seq)
        random.shuffle(shuffled)
        sequences.append(''.join(shuffled))
    return sequences

def write_fasta(sequences, output_path):
    """Write sequences to a fasta file."""
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">random_seq_{i+1:03d}\n")
            f.write(f"{seq}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate randomized protein sequences")
    parser.add_argument("--input", type=str, default="examples/inputs/5y0m.fasta",
                        help="Input fasta file")
    parser.add_argument("--output", type=str, default="examples/inputs/random_100.fasta",
                        help="Output fasta file")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of random sequences to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Read original sequence
    original_seq = read_fasta(args.input)
    print(f"Original sequence length: {len(original_seq)}")

    # Generate random sequences
    sequences = generate_random_sequences(original_seq, n=args.n, seed=args.seed)
    print(f"Generated {len(sequences)} random sequences")

    # Write to output file
    write_fasta(sequences, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()