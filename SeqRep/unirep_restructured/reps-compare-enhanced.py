#!/usr/bin/env python3
# reps_compare_fixed.py
import h5py
import numpy as np
import argparse
import os
import sys

def load_h5_file(file_path):
    """Load representations from H5 file."""
    try:
        print(f"[INFO] Loading file: {file_path}")
        data = {}

        with h5py.File(file_path, 'r') as f:
            # Print the structure of the file for debugging
            print(f"[DEBUG] File structure:")
            print_h5_structure(f)

            # Handle the file structure based on what we find
            for key in f.keys():
                # Check if it's a Group or Dataset
                item = f[key]

                if isinstance(item, h5py.Group):
                    # It's a group - handle nested structure
                    data[key] = {}
                    for subkey in item.keys():
                        data[key][subkey] = item[subkey][()]
                else:
                    # It's a dataset - store directly
                    data[key] = item[()]

        return data
    except Exception as e:
        print(f"[ERROR] Failed to load file {file_path}: {e}")
        return None

def print_h5_structure(h5obj, indent=0):
    """Recursively print the structure of an H5 file/group."""
    for key in h5obj.keys():
        item = h5obj[key]
        if isinstance(item, h5py.Group):
            print(" " * indent + f"- {key} (Group)")
            print_h5_structure(item, indent + 2)
        else:
            try:
                shape = item.shape
                dtype = item.dtype
                print(" " * indent + f"- {key} (Dataset)")
                print(" " * (indent + 2) + f"Shape: {shape}, Type: {dtype}")
            except:
                print(" " * indent + f"- {key} (Unknown/Error)")

def compare_representations(rep1_data, rep2_data):
    """Compare representations from two sources."""
    results = {}

    # Find common keys (could be either sequences or representation types)
    common_keys = set(rep1_data.keys()) & set(rep2_data.keys())

    for key in common_keys:
        item1 = rep1_data[key]
        item2 = rep2_data[key]

        # Case 1: Both are arrays (direct datasets)
        if isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            # Check shapes
            if item1.shape != item2.shape:
                results[key] = {
                    'match': False,
                    'reason': f"Shape mismatch: {item1.shape} vs {item2.shape}"
                }
                continue

            # Compute statistics
            abs_diff = np.abs(item1 - item2)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            std_diff = np.std(abs_diff)

            # Determine if they match (using a reasonable tolerance)
            match = max_diff < 1e-5

            results[key] = {
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'std_diff': std_diff
            }

        # Case 2: Both are dictionaries (nested structures)
        elif isinstance(item1, dict) and isinstance(item2, dict):
            results[key] = {}
            common_subkeys = set(item1.keys()) & set(item2.keys())

            for subkey in common_subkeys:
                subitem1 = item1[subkey]
                subitem2 = item2[subkey]

                # Check shapes
                if subitem1.shape != subitem2.shape:
                    results[key][subkey] = {
                        'match': False,
                        'reason': f"Shape mismatch: {subitem1.shape} vs {subitem2.shape}"
                    }
                    continue

                # Compute statistics
                abs_diff = np.abs(subitem1 - subitem2)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                std_diff = np.std(abs_diff)

                # Determine if they match (using a reasonable tolerance)
                match = max_diff < 1e-5

                results[key][subkey] = {
                    'match': match,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff
                }

        # Case 3: Mismatched types
        else:
            results[key] = {
                'match': False,
                'reason': f"Type mismatch: {type(item1)} vs {type(item2)}"
            }

    # Report missing keys
    missing_in_rep1 = set(rep2_data.keys()) - set(rep1_data.keys())
    missing_in_rep2 = set(rep1_data.keys()) - set(rep2_data.keys())

    if missing_in_rep1:
        results['__missing_in_rep1__'] = list(missing_in_rep1)

    if missing_in_rep2:
        results['__missing_in_rep2__'] = list(missing_in_rep2)

    return results

def print_results(comparison, verbose=False):
    """Print comparison results in a readable format."""
    print("\n[INFO] Comparison Results:")
    all_match = True

    for key, result in comparison.items():
        # Skip special keys
        if key in ('__missing_in_rep1__', '__missing_in_rep2__'):
            continue

        # Case 1: Simple comparison for a single item
        if isinstance(result, dict) and 'match' in result:
            match = result['match']
            all_match = all_match and match

            if match:
                print(f"  {key}: MATCH")
            else:
                print(f"  {key}: NO MATCH")
                if 'reason' in result:
                    print(f"    Reason: {result['reason']}")
                else:
                    print(f"    Max difference: {result['max_diff']}")

            if verbose and 'max_diff' in result:
                print(f"    Max difference: {result['max_diff']}")
                print(f"    Mean difference: {result['mean_diff']}")
                print(f"    Std difference: {result['std_diff']}")

        # Case 2: Nested comparison for a group
        elif isinstance(result, dict):
            print(f"  {key}:")
            item_match = True

            for subkey, subresult in result.items():
                if not isinstance(subresult, dict) or 'match' not in subresult:
                    continue

                sub_match = subresult['match']
                item_match = item_match and sub_match

                if sub_match:
                    print(f"    {subkey}: MATCH")
                else:
                    print(f"    {subkey}: NO MATCH")
                    if 'reason' in subresult:
                        print(f"      Reason: {subresult['reason']}")
                    else:
                        print(f"      Max difference: {subresult['max_diff']}")

                if verbose and 'max_diff' in subresult:
                    print(f"      Max difference: {subresult['max_diff']}")
                    print(f"      Mean difference: {subresult['mean_diff']}")
                    print(f"      Std difference: {subresult['std_diff']}")

            all_match = all_match and item_match

    # Report missing keys
    if '__missing_in_rep1__' in comparison:
        print("\n[INFO] Keys present in file2 but missing in file1:")
        for key in comparison['__missing_in_rep1__']:
            print(f"  - {key}")
        all_match = False

    if '__missing_in_rep2__' in comparison:
        print("\n[INFO] Keys present in file1 but missing in file2:")
        for key in comparison['__missing_in_rep2__']:
            print(f"  - {key}")
        all_match = False

    # Final result
    print("\n[INFO] Final Result:")
    if all_match:
        print("  ALL REPRESENTATIONS MATCH (within tolerance)")
        return 0
    else:
        print("  SOME REPRESENTATIONS DO NOT MATCH")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Compare UniRep representations")
    parser.add_argument("file1", help="First H5 representation file")
    parser.add_argument("file2", help="Second H5 representation file")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                       help="Tolerance for numerical differences")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed comparison results")

    args = parser.parse_args()

    # Check if files exist
    for file_path in [args.file1, args.file2]:
        if not os.path.exists(file_path):
            print(f"[ERROR] File does not exist: {file_path}")
            return 1

    # Load representations
    print("[INFO] Loading representation files...")
    rep1_data = load_h5_file(args.file1)
    rep2_data = load_h5_file(args.file2)

    if rep1_data is None or rep2_data is None:
        print("[ERROR] Failed to load one or both of the representation files")
        return 1

    # Compare representations
    print("[INFO] Comparing representations...")
    comparison = compare_representations(rep1_data, rep2_data)

    # Print results
    return print_results(comparison, args.verbose)

if __name__ == "__main__":
    sys.exit(main())