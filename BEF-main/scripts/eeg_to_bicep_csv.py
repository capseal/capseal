#!/usr/bin/env python3
"""Convert BEF EEG npy/json outputs into BICEP-style sequence CSV + predictions."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _flatten_single(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return _flatten_single(value[0])
    return float(value)


def load_bef_json(path: Path) -> Dict:
    with path.open() as f:
        data = json.load(f)
    return data


def attention_stats(weights: List[float]) -> Tuple[float, float, int]:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0
    arr = np.clip(arr, 1e-9, None)
    arr = arr / arr.sum()
    entropy = float(-(arr * np.log(arr)).sum())
    idx = int(arr.argmax())
    return entropy, float(arr[idx]), idx


def derive_label_from_path(path: Path) -> int:
    name = path.name.lower()
    if 'funwithfractals' in name or 'task-funwithfractals' in name:
        return 1
    if 'resting' in name:
        return 0
    # default fallback: use presence of task marker
    return 0


def build_sequences(npy_paths: List[Path], npy_root: Path, meta_root: Path,
                    window: int, hop: int, steps_per_sequence: int,
                    seq_out: Path) -> Tuple[pd.DataFrame, Dict[int, Path], Dict[int, Dict[str, float]]]:
    rows = []
    seq_paths: Dict[int, Path] = {}
    final_stats: Dict[int, Dict[str, float]] = {}
    sequence_id = 0
    for npy_path in sorted(npy_paths):
        rel = npy_path.relative_to(npy_root)
        meta_path = meta_root / rel.with_suffix('.json')
        if not meta_path.exists():
            print(f"[WARN] Missing meta for {npy_path}, skipping")
            continue
        label = derive_label_from_path(npy_path)
        data = np.load(npy_path)
        if data.ndim == 3:
            data = data[0]
        channels, samples = data.shape
        steps = max(1, (samples - window) // hop + 1)
        seq_paths[sequence_id] = npy_path
        chunk_id = 0
        for start_step in range(0, steps, steps_per_sequence):
            seq_paths[sequence_id] = npy_path
            sub_steps = min(steps_per_sequence, steps - start_step)
            for local_idx in range(sub_steps):
                global_step = start_step + local_idx
                start = global_step * hop
                end = start + window
                if end > samples:
                    start = max(0, samples - window)
                    end = start + window
                window_data = data[:, start:end]
                flat = window_data.reshape(-1)
                state_mean = float(flat.mean())
                state_std = float(flat.std())
                q10 = float(np.percentile(flat, 10))
                q90 = float(np.percentile(flat, 90))
                aleatoric = float(window_data.var())
                epistemic = float(window_data.mean(axis=1).std())
                rows.append({
                    'sequence_id': sequence_id,
                    'step': local_idx,
                    'input': state_mean,
                    'state_0': state_mean,
                    'state_mean': state_mean,
                    'state_std': state_std,
                    'state_q10': q10,
                    'state_q90': q90,
                    'aleatoric_unc': aleatoric,
                    'epistemic_unc': epistemic,
                    'target': label,
                    'source_file': f"{npy_path}#chunk{chunk_id}",
                })
                last_stats = {
                    'state_mean': state_mean,
                    'state_std': state_std,
                    'state_q10': q10,
                    'state_q90': q90,
                    'aleatoric_unc': aleatoric,
                    'epistemic_unc': epistemic,
                }
            final_stats[sequence_id] = last_stats
            chunk_id += 1
            sequence_id += 1
    df = pd.DataFrame(rows)
    df.to_csv(seq_out, index=False)
    return df, seq_paths, final_stats


def build_predictions(seq_paths: Dict[int, Path], npy_root: Path, bef_root: Path,
                      final_stats: Dict[int, Dict[str, float]], pred_out: Path) -> None:
    rows = []
    seq_ids = sorted(seq_paths.keys())
    for seq_id in seq_ids:
        npy_path = seq_paths.get(seq_id)
        if npy_path is None:
            continue
        rel = npy_path.relative_to(npy_root)
        bef_path = bef_root / rel.with_suffix('.json')
        if not bef_path.exists():
            print(f"[WARN] Missing BEF output for {npy_path}, skipping")
            continue
        bef = load_bef_json(bef_path)
        pred = _flatten_single(bef.get('prediction', 0.5))
        aleatoric = _flatten_single(bef.get('aleatoric_uncertainty', 0.0))
        epistemic = _flatten_single(bef.get('epistemic_uncertainty', 0.0))
        total_unc = aleatoric + epistemic
        confidence = 1.0 / (1.0 + max(1e-6, total_unc))
        weights = bef.get('attention_weights', [])
        weights = weights[0] if isinstance(weights, list) and weights and isinstance(weights[0], list) else weights
        entropy, attn_max, attn_arg = attention_stats(weights)
        stats = final_stats.get(seq_id)
        if not stats:
            continue
        rows.append({
            'sequence_id': seq_id,
            'final_prediction': pred,
            'target': derive_label_from_path(npy_path),
            'confidence': confidence,
            'state_mean': stats['state_mean'],
            'state_std': stats['state_std'],
            'state_q10': stats['state_q10'],
            'state_q90': stats['state_q90'],
            'aleatoric_unc': stats['aleatoric_unc'],
            'epistemic_unc': stats['epistemic_unc'],
            'attention_entropy': entropy,
            'collapse_temperature': 1.0,
            'attention_max': attn_max,
            'attention_argmax': attn_arg,
        })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_out, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert EEG npy/Bef outputs into BICEP-compatible CSVs')
    parser.add_argument('--npy-root', default='data/bef_runs/npy')
    parser.add_argument('--meta-root', default='data/bef_runs/meta')
    parser.add_argument('--bef-root', default='data/bef_runs/bef')
    parser.add_argument('--seq-out', default='data/eeg_sequences.csv')
    parser.add_argument('--pred-out', default='data/eeg_predictions.csv')
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--hop', type=int, default=100)
    parser.add_argument('--sequence-steps', type=int, default=60,
                        help='Number of windows per generated sequence (set smaller to create multiple sequences per file)')
    args = parser.parse_args()

    global npy_root
    npy_root = Path(args.npy_root)
    meta_root = Path(args.meta_root)
    bef_root = Path(args.bef_root)
    seq_out = Path(args.seq_out)
    pred_out = Path(args.pred_out)

    npy_paths = sorted(npy_root.rglob('*.npy'))
    if not npy_paths:
        raise SystemExit('No npy files found')

    seq_df, seq_paths, final_stats = build_sequences(
        npy_paths, npy_root, meta_root,
        args.window, args.hop, args.sequence_steps,
        seq_out)
    build_predictions(seq_paths, npy_root, bef_root, final_stats, pred_out)
    print(f"Wrote sequences to {seq_out} ({len(seq_df)} rows)")
    print(f"Wrote predictions to {pred_out}")
