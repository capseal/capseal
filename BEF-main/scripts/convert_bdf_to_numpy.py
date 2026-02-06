#!/usr/bin/env python3
"""Convert BioSemi BDF recordings to NumPy tensors for BEF EEG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import pyedflib  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "pyedflib is required for BDF conversion. Install it via `pip install pyedflib`."
    ) from exc


def _resolve_channels(reader: pyedflib.EdfReader, picks: Sequence[str] | None) -> List[int]:
    labels = reader.getSignalLabels()
    if not picks:
        return list(range(len(labels)))
    indices: List[int] = []
    lower_map = {label.lower(): idx for idx, label in enumerate(labels)}
    for name in picks:
        key = name.lower()
        if key not in lower_map:
            raise ValueError(f"Channel '{name}' not found in BDF labels: {labels}")
        indices.append(lower_map[key])
    return indices


def convert_bdf(
    bdf_path: Path,
    out_path: Path,
    meta_path: Path | None = None,
    picks: Sequence[str] | None = None,
    start: float = 0.0,
    duration: float | None = None,
    dtype: str = "float32",
) -> dict:
    reader = pyedflib.EdfReader(str(bdf_path))
    try:
        channel_indices = _resolve_channels(reader, picks)
        labels = reader.getSignalLabels()
        sfreqs = reader.getSampleFrequencies()

        channel_data: List[np.ndarray] = []
        min_len = None
        for idx in channel_indices:
            signal = reader.readSignal(idx)
            sr = float(sfreqs[idx]) if isinstance(sfreqs, np.ndarray) else float(sfreqs[idx])
            start_idx = int(start * sr)
            stop_idx = int((start + duration) * sr) if duration else signal.shape[0]
            start_idx = max(0, min(start_idx, signal.shape[0]))
            stop_idx = max(start_idx, min(stop_idx, signal.shape[0]))
            segment = signal[start_idx:stop_idx]
            channel_data.append(segment.astype(dtype))
            if min_len is None or len(segment) < min_len:
                min_len = len(segment)
        if min_len is None:
            raise RuntimeError("No channels selected for conversion")

        stacked = np.stack([seg[:min_len] for seg in channel_data], axis=0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, stacked)

        metadata = {
            "source_bdf": str(bdf_path),
            "output_numpy": str(out_path),
            "channel_labels": [labels[i] for i in channel_indices],
            "sample_frequencies": [float(sfreqs[i]) for i in channel_indices],
            "start_seconds": start,
            "duration_seconds": duration if duration is not None else len(stacked[0]) / float(sfreqs[channel_indices[0]]),
            "dtype": dtype,
        }
        if meta_path:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata
    finally:
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert BDF EEG recordings to NumPy tensors")
    parser.add_argument("--bdf", required=True, help="Path to input .bdf file")
    parser.add_argument("--out", help="Output .npy path (defaults to <bdf>.npy)")
    parser.add_argument("--meta", help="Optional JSON metadata path")
    parser.add_argument("--channels", nargs="*", help="Subset of channel names to include")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, help="Duration in seconds")
    parser.add_argument("--dtype", default="float32", help="Output dtype (default float32)")
    return parser


def main():
    args = build_parser().parse_args()
    bdf = Path(args.bdf)
    if not bdf.exists():
        raise SystemExit(f"BDF file not found: {bdf}")
    out = Path(args.out) if args.out else bdf.with_suffix(".npy")
    meta = Path(args.meta) if args.meta else bdf.with_suffix(".json")
    metadata = convert_bdf(
        bdf_path=bdf,
        out_path=out,
        meta_path=meta,
        picks=args.channels,
        start=args.start,
        duration=args.duration,
        dtype=args.dtype,
    )
    print(f"Saved NumPy tensor to {out} (shape={metadata['channel_labels']} x duration)")
    if meta:
        print(f"Metadata written to {meta}")


if __name__ == "__main__":
    main()
