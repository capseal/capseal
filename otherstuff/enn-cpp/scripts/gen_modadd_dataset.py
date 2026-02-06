#!/usr/bin/env python3
"""Generate modular addition datasets for grokking experiments (Path A: multi-bit BCE).

This script creates CSV datasets for the modular addition task: (a + b) mod p.
The target is encoded as binary bits, enabling BCE-per-bit training without
requiring softmax/cross-entropy.

This is the canonical grokking benchmark where:
- Training accuracy goes high early
- Test accuracy stays low for a long time
- Then test accuracy suddenly jumps (grokking!)

Examples:
    # Generate p=31 (5-bit) modular addition
    python gen_modadd_dataset.py --prime 31 --train_frac 0.5 --output /tmp/modadd_p31.csv

    # Generate p=97 (7-bit) for classic grokking setup
    python gen_modadd_dataset.py --prime 97 --train_frac 0.3 --output /tmp/modadd_p97.csv

    # Verify dataset
    python gen_modadd_dataset.py --verify /tmp/modadd_p31.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import math

import numpy as np


def int_to_bits(value: int, n_bits: int) -> List[int]:
    """Convert integer to binary representation (LSB first)."""
    return [(value >> i) & 1 for i in range(n_bits)]


def bits_to_int(bits: List[int]) -> int:
    """Convert binary representation (LSB first) to integer."""
    return sum(b << i for i, b in enumerate(bits))


def required_bits(prime: int) -> int:
    """Number of bits needed to represent values in [0, prime-1]."""
    return max(1, math.ceil(math.log2(prime)))


def generate_modadd_data(
    prime: int,
    train_frac: float,
    seed: int,
) -> Tuple[List[dict], List[dict], int]:
    """Generate modular addition dataset with train/test split.

    Args:
        prime: The modulus p
        train_frac: Fraction of data for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_rows, test_rows, n_bits)
    """
    rng = np.random.default_rng(seed)
    n_bits = required_bits(prime)

    # Generate all p^2 pairs
    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    n_total = len(all_pairs)
    n_train = int(n_total * train_frac)

    # Shuffle and split
    indices = rng.permutation(n_total)
    train_indices = set(indices[:n_train])

    train_rows = []
    test_rows = []

    for idx, (a, b) in enumerate(all_pairs):
        c = (a + b) % prime
        target_bits = int_to_bits(c, n_bits)

        rows = generate_sequence_rows(
            sequence_id=idx,
            a=a,
            b=b,
            target_bits=target_bits,
            n_bits=n_bits,
            prime=prime
        )

        if idx in train_indices:
            train_rows.extend(rows)
        else:
            test_rows.extend(rows)

    return train_rows, test_rows, n_bits


def generate_sequence_rows(
    sequence_id: int,
    a: int,
    b: int,
    target_bits: List[int],
    n_bits: int,
    prime: int
) -> List[dict]:
    """Generate CSV rows for a single (a, b) -> (a+b) mod p example.

    We encode as a 2-step sequence:
    - Step 0: Input bits of 'a'
    - Step 1: Input bits of 'b'
    - Target: bits of (a+b) mod p (same for both steps, final step matters)

    The input encoding uses the binary bits of each operand, plus sinusoidal
    position encoding to help the model distinguish operands.

    Args:
        sequence_id: Unique identifier
        a, b: The two operands
        target_bits: Binary representation of (a+b) mod p
        n_bits: Number of bits
        prime: The modulus

    Returns:
        List of 2 row dicts
    """
    rows = []

    for step, value in enumerate([a, b]):
        input_bits = int_to_bits(value, n_bits)

        # Base row with metadata
        row = {
            'sequence_id': sequence_id,
            'step': step,
            'operand_a': a,
            'operand_b': b,
            'result': bits_to_int(target_bits),
        }

        # Input features: binary bits encoded as -1/+1
        for i, bit in enumerate(input_bits):
            row[f'input_bit_{i}'] = 1.0 if bit == 1 else -1.0

        # Add position encoding to distinguish step 0 (a) from step 1 (b)
        row['pos_sin'] = np.sin(np.pi * step)
        row['pos_cos'] = np.cos(np.pi * step)

        # Target bits (same for both steps, but only final step matters with --predict_final_only)
        for i, bit in enumerate(target_bits):
            row[f'target_bit_{i}'] = float(bit)

        # Standard columns expected by bicep_to_enn (using first input bit as primary)
        row['input'] = row['input_bit_0']
        row['state_mean'] = row['input_bit_0']
        row['state_std'] = 0.1
        row['state_q10'] = row['input_bit_0'] - 0.1
        row['state_q90'] = row['input_bit_0'] + 0.1
        row['aleatoric_unc'] = 0.01
        row['epistemic_unc'] = 0.01
        # Legacy single target (first bit, for compatibility)
        row['target'] = row['target_bit_0']

        rows.append(row)

    return rows


def write_csv(rows: List[dict], output_path: str) -> None:
    """Write rows to CSV file."""
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_csv(csv_path: str) -> Tuple[bool, str]:
    """Verify a modular addition dataset."""
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            return False, "No rows found"

        # Find n_bits from column names
        n_bits = 0
        for col in rows[0].keys():
            if col.startswith('target_bit_'):
                idx = int(col.split('_')[-1])
                n_bits = max(n_bits, idx + 1)

        if n_bits == 0:
            return False, "No target_bit columns found"

        # Group by sequence
        sequences: Dict[int, List[dict]] = {}
        for row in rows:
            seq_id = int(row['sequence_id'])
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(row)

        # Verify each sequence
        errors = []
        for seq_id, seq_rows in sequences.items():
            if len(seq_rows) != 2:
                errors.append(f"Sequence {seq_id}: expected 2 rows, got {len(seq_rows)}")
                continue

            a = int(seq_rows[0]['operand_a'])
            b = int(seq_rows[0]['operand_b'])
            stored_result = int(seq_rows[0]['result'])

            # Reconstruct target from bits
            target_bits = [int(float(seq_rows[0][f'target_bit_{i}'])) for i in range(n_bits)]
            reconstructed = bits_to_int(target_bits)

            if reconstructed != stored_result:
                errors.append(f"Sequence {seq_id}: bit reconstruction mismatch")

            # We can't verify without knowing prime, but we can check consistency
            if seq_rows[0]['operand_a'] != seq_rows[1]['operand_a']:
                errors.append(f"Sequence {seq_id}: operand_a mismatch between steps")

        if errors:
            return False, f"{len(errors)} errors:\n" + "\n".join(errors[:10])

        n_seqs = len(sequences)
        return True, f"Verified {n_seqs} sequences, {n_bits} output bits"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate modular addition datasets for grokking (multi-bit BCE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # p=31 (5-bit output), 50% train split
  python gen_modadd_dataset.py --prime 31 --train_frac 0.5 --output /tmp/modadd_p31.csv

  # p=97 (7-bit output), classic grokking setup
  python gen_modadd_dataset.py --prime 97 --train_frac 0.3 --output /tmp/modadd_p97.csv

  # Verify existing dataset
  python gen_modadd_dataset.py --verify /tmp/modadd_p31.csv

Output format:
  - 2 steps per sequence (step 0 = operand a, step 1 = operand b)
  - Input features: input_bit_0..input_bit_{n-1} (±1 encoding)
  - Target features: target_bit_0..target_bit_{n-1} (0/1 for BCE)
  - Use --predict_final_only to train only on step 1 predictions
"""
    )

    parser.add_argument('--prime', type=int, default=31,
                        help='Modulus p (default: 31, needs 5 bits)')
    parser.add_argument('--train_frac', type=float, default=0.5,
                        help='Fraction for training (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')
    parser.add_argument('--verify', type=str, metavar='CSV_PATH',
                        help='Verify existing CSV instead of generating')

    args = parser.parse_args()

    if args.verify:
        success, msg = verify_csv(args.verify)
        print(msg)
        sys.exit(0 if success else 1)

    if args.output is None:
        parser.error("--output is required for generation")

    n_bits = required_bits(args.prime)
    print(f"Prime {args.prime} requires {n_bits} bits for output encoding")

    # Generate data
    train_rows, test_rows, n_bits = generate_modadd_data(
        prime=args.prime,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    # Combine and write
    all_rows = train_rows + test_rows
    write_csv(all_rows, args.output)

    # Also write split files
    base = Path(args.output).stem
    parent = Path(args.output).parent
    write_csv(train_rows, str(parent / f"{base}_train.csv"))
    write_csv(test_rows, str(parent / f"{base}_test.csv"))

    n_train_seqs = len(train_rows) // 2
    n_test_seqs = len(test_rows) // 2
    n_total = args.prime ** 2

    print(f"\nGenerated modular addition dataset:")
    print(f"  prime: {args.prime}")
    print(f"  output_bits: {n_bits}")
    print(f"  total pairs: {n_total}")
    print(f"  train sequences: {n_train_seqs} ({100*n_train_seqs/n_total:.1f}%)")
    print(f"  test sequences: {n_test_seqs} ({100*n_test_seqs/n_total:.1f}%)")
    print(f"  seed: {args.seed}")
    print(f"  files:")
    print(f"    combined: {args.output}")
    print(f"    train: {parent / f'{base}_train.csv'}")
    print(f"    test: {parent / f'{base}_test.csv'}")

    # Print example
    print(f"\nExample sequence (a=3, b=5, p={args.prime}):")
    c = (3 + 5) % args.prime
    bits = int_to_bits(c, n_bits)
    print(f"  (3 + 5) mod {args.prime} = {c}")
    print(f"  binary (LSB first): {bits}")
    print(f"  input_bit columns: ±1 encoding of operand bits")
    print(f"  target_bit columns: 0/1 for BCE training")


if __name__ == "__main__":
    main()
