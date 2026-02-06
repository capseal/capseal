#!/usr/bin/env python3
"""Run deterministic pipeline twice and ensure outputs match bit-for-bit."""

import argparse
import filecmp
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_apps(bicep_csv: Path, telemetry_out: Path) -> None:
    enn_root = REPO_ROOT / "enn-cpp"
    binary = enn_root / "apps" / "bicep_to_enn"
    if not binary.exists():
        subprocess.run(["make", "apps/bicep_to_enn"], cwd=enn_root, check=True)
    subprocess.run(
        [str(binary), str(bicep_csv), "--telemetry", str(telemetry_out)],
        cwd=enn_root,
        check=True,
    )


def run_q_dump(output: Path) -> None:
    script = REPO_ROOT / "FusionAlpha" / "python" / "dump_q.py"
    subprocess.run(
        ["python", str(script), "--output", str(output)],
        cwd=REPO_ROOT / "FusionAlpha",
        check=True,
    )


def compare_files(a: Path, b: Path) -> bool:
    return filecmp.cmp(str(a), str(b), shallow=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay deterministic pipeline and compare outputs")
    parser.add_argument("--csv", default=str(Path.home() / "sample_bicep_enn_demo.csv"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    enn_root = REPO_ROOT / "enn-cpp"
    subprocess.run(["make", "deterministic"], cwd=enn_root, check=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        telemetry_a = tmp / "telemetry_a.csv"
        telemetry_b = tmp / "telemetry_b.csv"
        q_a = tmp / "q_a.json"
        q_b = tmp / "q_b.json"

        run_apps(csv_path, telemetry_a)
        run_q_dump(q_a)

        run_apps(csv_path, telemetry_b)
        run_q_dump(q_b)

        telemetry_match = compare_files(telemetry_a, telemetry_b)
        q_match = compare_files(q_a, q_b)

        print(f"Telemetry match: {telemetry_match}")
        print(f"Q-values match: {q_match}")

        if not (telemetry_match and q_match):
            raise SystemExit("Replay test failed: outputs differ")

        print("Replay test succeeded")


if __name__ == "__main__":
    main()
