#!/usr/bin/env python3
"""Generate reproducible parity datasets with redundant features for ENN training.

This script creates CSV datasets for the parity problem (XOR of all bits),
using a "redundant features" format that has been shown to work well with
Epistemic Neural Networks (ENN).

The key insight is that redundant features (state_mean = input, state_q10 = input - 1,
state_q90 = input + 1) help ENN learn the parity function more effectively.

Examples:
    # Generate full enumeration for 8-bit parity
    python gen_parity_dataset.py --n_bits 8 --mode full --output /tmp/parity_n8.csv

    # Generate sparse random samples
    python gen_parity_dataset.py --n_bits 16 --n_sequences 1000 --mode sparse --seed 42 --output /tmp/parity_sparse.csv

    # Verify an existing dataset
    python gen_parity_dataset.py --verify /tmp/parity_n8.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def compute_parity(bits: List[int]) -> int:
    """Compute XOR parity of a list of bits.

    Args:
        bits: List of integers (0 or 1)

    Returns:
        0 if even number of 1s, 1 if odd number of 1s
    """
    parity = 0
    for b in bits:
        parity ^= b
    return parity


def int_to_bits(value: int, n_bits: int) -> List[int]:
    """Convert an integer to a list of bits (MSB first).

    Args:
        value: Non-negative integer to convert
        n_bits: Number of bits to use

    Returns:
        List of n_bits integers (0 or 1)
    """
    bits = []
    for i in range(n_bits - 1, -1, -1):
        bits.append((value >> i) & 1)
    return bits


def bit_to_input(bit: int) -> float:
    """Convert a bit (0/1) to input encoding (+1.0/-1.0).

    We use +1.0 for bit=1 and -1.0 for bit=0, which is a common
    encoding that centers the data around zero.

    Args:
        bit: 0 or 1

    Returns:
        1.0 if bit is 1, -1.0 if bit is 0
    """
    return 1.0 if bit == 1 else -1.0


def generate_sequence_rows(
    sequence_id: int,
    bits: List[int],
    redundant: bool = True
) -> List[dict]:
    """Generate CSV rows for a single bit sequence.

    Each bit in the sequence becomes one row/step. The target (parity)
    is the same for all steps in a sequence.

    Args:
        sequence_id: Unique identifier for this sequence
        bits: List of bits for this sequence
        redundant: Whether to use redundant features format

    Returns:
        List of dicts, one per step, ready for CSV writing
    """
    target = compute_parity(bits)
    rows = []

    for step, bit in enumerate(bits):
        input_val = bit_to_input(bit)

        if redundant:
            # Redundant features format - creates correlated features
            # that help ENN learn the structure
            row = {
                'sequence_id': sequence_id,
                'step': step,
                'input': input_val,
                'state_mean': input_val,           # Redundant: same as input
                'state_std': 1.0,                  # Constant std
                'state_q10': input_val - 1.0,      # Redundant: shifted down
                'state_q90': input_val + 1.0,      # Redundant: shifted up
                'aleatoric_unc': 0.1,              # Small constant
                'epistemic_unc': 0.1,              # Small constant
                'target': target
            }
        else:
            # Minimal format - just input and target
            row = {
                'sequence_id': sequence_id,
                'step': step,
                'input': input_val,
                'target': target
            }

        rows.append(row)

    return rows


def generate_full_enumeration(n_bits: int, redundant: bool = True) -> List[dict]:
    """Generate all 2^n_bits possible sequences.

    Sequences are ordered by their integer value (sequence_id = integer value).
    This provides deterministic, reproducible ordering.

    Args:
        n_bits: Number of bits per sequence
        redundant: Whether to use redundant features format

    Returns:
        List of all rows for all sequences
    """
    all_rows = []
    n_sequences = 2 ** n_bits

    for seq_id in range(n_sequences):
        bits = int_to_bits(seq_id, n_bits)
        rows = generate_sequence_rows(seq_id, bits, redundant)
        all_rows.extend(rows)

    return all_rows


def generate_sparse_sample(
    n_bits: int,
    n_sequences: int,
    seed: int,
    redundant: bool = True
) -> List[dict]:
    """Generate a random sparse sample of sequences.

    Randomly samples n_sequences from the 2^n_bits possible sequences
    without replacement (if n_sequences <= 2^n_bits).

    Args:
        n_bits: Number of bits per sequence
        n_sequences: Number of sequences to sample
        seed: Random seed for reproducibility
        redundant: Whether to use redundant features format

    Returns:
        List of all rows for sampled sequences
    """
    rng = np.random.default_rng(seed)
    max_sequences = 2 ** n_bits

    if n_sequences > max_sequences:
        print(f"Warning: Requested {n_sequences} sequences but only {max_sequences} "
              f"possible with {n_bits} bits. Generating full enumeration instead.",
              file=sys.stderr)
        return generate_full_enumeration(n_bits, redundant)

    # Sample without replacement
    sampled_values = rng.choice(max_sequences, size=n_sequences, replace=False)
    sampled_values = np.sort(sampled_values)  # Sort for deterministic ordering

    all_rows = []
    for idx, value in enumerate(sampled_values):
        bits = int_to_bits(int(value), n_bits)
        # Use idx as sequence_id for contiguous IDs in sparse mode
        rows = generate_sequence_rows(idx, bits, redundant)
        all_rows.extend(rows)

    return all_rows


def write_csv(rows: List[dict], output_path: str) -> None:
    """Write rows to a CSV file.

    Args:
        rows: List of dicts with consistent keys
        output_path: Path to output CSV file
    """
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_label_balance(rows: List[dict]) -> float:
    """Compute the fraction of sequences with target=1.

    Args:
        rows: List of row dicts

    Returns:
        Fraction of unique sequences with target=1
    """
    # Get unique sequence targets
    seq_targets = {}
    for row in rows:
        seq_id = row['sequence_id']
        if seq_id not in seq_targets:
            seq_targets[seq_id] = row['target']

    if not seq_targets:
        return 0.0

    return sum(seq_targets.values()) / len(seq_targets)


def verify_csv(csv_path: str) -> Tuple[bool, str]:
    """Verify that a parity dataset CSV has correct labels.

    Reads the CSV, reconstructs the bit sequences from the input encodings,
    recomputes the parity labels, and checks they match.

    Args:
        csv_path: Path to CSV file to verify

    Returns:
        Tuple of (success, message)
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)

            if 'sequence_id' not in reader.fieldnames:
                return False, "CSV missing 'sequence_id' column"
            if 'step' not in reader.fieldnames:
                return False, "CSV missing 'step' column"
            if 'input' not in reader.fieldnames:
                return False, "CSV missing 'input' column"
            if 'target' not in reader.fieldnames:
                return False, "CSV missing 'target' column"

            # Group rows by sequence
            sequences = {}
            for row in reader:
                seq_id = int(row['sequence_id'])
                step = int(row['step'])
                input_val = float(row['input'])
                target = int(row['target'])

                if seq_id not in sequences:
                    sequences[seq_id] = {'steps': {}, 'target': target}

                sequences[seq_id]['steps'][step] = input_val

                # Check target consistency within sequence
                if sequences[seq_id]['target'] != target:
                    return False, f"Inconsistent target in sequence {seq_id}"

            if not sequences:
                return False, "No sequences found in CSV"

            # Verify each sequence
            n_correct = 0
            n_total = len(sequences)
            errors = []

            for seq_id, seq_data in sequences.items():
                steps = seq_data['steps']
                target = seq_data['target']

                # Reconstruct bits from inputs
                n_bits = len(steps)
                bits = []
                for step in range(n_bits):
                    if step not in steps:
                        errors.append(f"Sequence {seq_id} missing step {step}")
                        continue
                    input_val = steps[step]
                    # input = 1.0 means bit=1, input = -1.0 means bit=0
                    bit = 1 if input_val > 0 else 0
                    bits.append(bit)

                # Compute expected parity
                expected_parity = compute_parity(bits)

                if expected_parity == target:
                    n_correct += 1
                else:
                    errors.append(
                        f"Sequence {seq_id}: bits={bits}, expected={expected_parity}, got={target}"
                    )

            if errors:
                error_summary = f"{len(errors)} errors found. First few:\n"
                error_summary += "\n".join(errors[:5])
                return False, error_summary

            return True, f"All {n_total} sequences verified correctly"

    except FileNotFoundError:
        return False, f"File not found: {csv_path}"
    except Exception as e:
        return False, f"Error reading CSV: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible parity datasets with redundant features for ENN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full enumeration of 8-bit parity (256 sequences)
  python gen_parity_dataset.py --n_bits 8 --mode full --output /tmp/parity_n8.csv

  # Sparse sampling of 16-bit parity (1000 random sequences)
  python gen_parity_dataset.py --n_bits 16 --n_sequences 1000 --mode sparse --seed 42 --output /tmp/parity_sparse.csv

  # Verify an existing dataset
  python gen_parity_dataset.py --verify /tmp/parity_n8.csv

  # Generate without redundant features
  python gen_parity_dataset.py --n_bits 8 --mode full --no-redundant --output /tmp/parity_minimal.csv
"""
    )

    # Generation options
    parser.add_argument(
        '--n_bits', type=int, default=8,
        help='Number of bits per sequence (default: 8)'
    )
    parser.add_argument(
        '--n_sequences', type=int, default=None,
        help='Number of sequences to generate (default: 2^n_bits for full mode)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: stdout summary only)'
    )
    parser.add_argument(
        '--mode', choices=['full', 'sparse'], default='full',
        help='Generation mode: full enumeration or sparse sampling (default: full)'
    )
    parser.add_argument(
        '--redundant', dest='redundant', action='store_true', default=True,
        help='Use redundant features format (default: True)'
    )
    parser.add_argument(
        '--no-redundant', dest='redundant', action='store_false',
        help='Disable redundant features (minimal format)'
    )

    # Verification mode
    parser.add_argument(
        '--verify', type=str, metavar='CSV_PATH',
        help='Verify an existing CSV file instead of generating'
    )

    args = parser.parse_args()

    # Verification mode
    if args.verify:
        print(f"Verifying: {args.verify}")
        success, message = verify_csv(args.verify)
        print(message)
        sys.exit(0 if success else 1)

    # Generation mode
    if args.output is None:
        parser.error("--output is required for generation (or use --verify for verification)")

    # Determine number of sequences
    if args.n_sequences is None:
        if args.mode == 'full':
            n_sequences = 2 ** args.n_bits
        else:
            parser.error("--n_sequences is required for sparse mode")
    else:
        n_sequences = args.n_sequences

    # Generate data
    if args.mode == 'full':
        rows = generate_full_enumeration(args.n_bits, args.redundant)
        actual_sequences = 2 ** args.n_bits
    else:
        rows = generate_sparse_sample(args.n_bits, n_sequences, args.seed, args.redundant)
        actual_sequences = min(n_sequences, 2 ** args.n_bits)

    # Write output
    write_csv(rows, args.output)

    # Compute and print summary
    label_balance = compute_label_balance(rows)

    print(f"Generated parity dataset:")
    print(f"  n_bits: {args.n_bits}")
    print(f"  sequences: {actual_sequences}")
    print(f"  mode: {args.mode}")
    print(f"  seed: {args.seed}")
    print(f"  redundant: {args.redundant}")
    print(f"  output: {args.output}")
    print(f"  total_rows: {len(rows)}")
    print(f"  label_balance: {label_balance:.3f}")


if __name__ == "__main__":
    main()
