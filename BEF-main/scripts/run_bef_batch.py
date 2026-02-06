#!/usr/bin/env python3
"""Batch convert BDF recordings and run the BEF EEG model."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from convert_bdf_to_numpy import convert_bdf
    from run_bef_eeg import run as run_bef_single
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Unable to import helper scripts. Run this tool via `python scripts/run_bef_batch.py` from the repo root."
    ) from exc

LOGGER = logging.getLogger("run_bef_batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class BefArgs:
    input: str
    checkpoint: Optional[str]
    output: str
    output_dim: int
    n_paths: int
    mc_samples: int
    use_multiscale: bool
    return_intermediates: bool
    cpu: bool


def find_bdf_files(root: Path, limit: Optional[int] = None) -> List[Path]:
    files = sorted(root.rglob("*.bdf"))
    if limit is not None:
        return files[:limit]
    return files


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_bef(args: BefArgs) -> dict:
    from argparse import Namespace

    ns = Namespace(
        input=args.input,
        checkpoint=args.checkpoint,
        output=args.output,
        output_dim=args.output_dim,
        n_paths=args.n_paths,
        mc_samples=args.mc_samples,
        use_multiscale=args.use_multiscale,
        return_intermediates=args.return_intermediates,
        cpu=args.cpu,
    )
    run_bef_single(ns)
    with Path(args.output).open() as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert BDF files and run BEF EEG inference")
    parser.add_argument("--bdf-root", required=True, help="Directory containing .bdf recordings")
    parser.add_argument("--checkpoint", default="v46_submission/c1_bef.pt", help="BEF checkpoint path")
    parser.add_argument("--out-dir", default="bef_batch", help="Directory for outputs (npy/json)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds to extract per file")
    parser.add_argument("--channels", nargs="*", help="Optional list of channel names to include")
    parser.add_argument("--limit", type=int, help="Limit number of BDF files for quick tests")
    parser.add_argument("--output-dim", type=int, default=1, help="BEF output dimension (1 for C1, 4 for C2)")
    parser.add_argument("--n-paths", type=int, default=32, help="BEF adaptive BICEP paths per SDE")
    parser.add_argument("--mc-samples", type=int, default=1, help="FusionAlpha MC dropout samples")
    parser.add_argument("--use-multiscale", action="store_true", help="Enable multiscale ENN encoder")
    parser.add_argument("--return-intermediates", action="store_true", help="Persist intermediate tensors")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--resume", action="store_true", help="Skip files whose JSON already exists")
    args = parser.parse_args()

    root = Path(args.bdf_root)
    out_dir = Path(args.out_dir)
    npy_dir = out_dir / "npy"
    meta_dir = out_dir / "meta"
    bef_dir = out_dir / "bef"

    files = find_bdf_files(root, args.limit)
    if not files:
        raise SystemExit(f"No .bdf files found under {root}")

    summary_rows: List[dict] = []

    for bdf_path in files:
        rel = bdf_path.relative_to(root)
        npy_path = npy_dir / rel.with_suffix(".npy")
        meta_path = meta_dir / rel.with_suffix(".json")
        bef_json = bef_dir / rel.with_suffix(".json")

        if args.resume and bef_json.exists():
            LOGGER.info("Skipping %s (existing output)", bdf_path)
            with bef_json.open() as fh:
                result = json.load(fh)
            summary_rows.append({
                "bdf": str(bdf_path),
                "npy": str(npy_path),
                "prediction": result.get('prediction', [[None]])[0][0],
                "aleatoric_uncertainty": result.get('aleatoric_uncertainty', [[None]])[0][0] if result.get('aleatoric_uncertainty') else None,
                "epistemic_uncertainty": result.get('epistemic_uncertainty', [[None]])[0][0] if result.get('epistemic_uncertainty') else None,
            })
            continue

        LOGGER.info("Converting %s", bdf_path)
        ensure_parent(npy_path)
        ensure_parent(meta_path)
        convert_bdf(
            bdf_path=bdf_path,
            out_path=npy_path,
            meta_path=meta_path,
            picks=args.channels,
            start=0.0,
            duration=args.duration,
        )

        ensure_parent(bef_json)
        LOGGER.info("Running BEF on %s", npy_path)
        result = run_bef(BefArgs(
            input=str(npy_path),
            checkpoint=args.checkpoint,
            output=str(bef_json),
            output_dim=args.output_dim,
            n_paths=args.n_paths,
            mc_samples=args.mc_samples,
            use_multiscale=args.use_multiscale,
            return_intermediates=args.return_intermediates,
            cpu=args.cpu,
        ))

        summary_rows.append({
            "bdf": str(bdf_path),
            "npy": str(npy_path),
            "prediction": result.get('prediction', [[None]])[0][0],
            "aleatoric_uncertainty": result.get('aleatoric_uncertainty', [[None]])[0][0] if result.get('aleatoric_uncertainty') else None,
            "epistemic_uncertainty": result.get('epistemic_uncertainty', [[None]])[0][0] if result.get('epistemic_uncertainty') else None,
        })

    summary_path = out_dir / "bef_summary.csv"
    ensure_parent(summary_path)
    with summary_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=["bdf", "npy", "prediction", "aleatoric_uncertainty", "epistemic_uncertainty"])
        writer.writeheader()
        writer.writerows(summary_rows)
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
