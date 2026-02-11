from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from capseal.cli.scan_cmd import scan_command
from capseal.risk_engine import evaluate_action_risk, evaluate_risk


def _write_model(workspace: Path, grid_idx: int, alpha_val: int = 9, beta_val: int = 3) -> Path:
    model_dir = workspace / ".capseal" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    alpha = np.ones(1024, dtype=np.int64)
    beta = np.ones(1024, dtype=np.int64)
    alpha[grid_idx] = alpha_val
    beta[grid_idx] = beta_val
    model_path = model_dir / "beta_posteriors.npz"
    np.savez(model_path, alpha=alpha, beta=beta, n_episodes=alpha_val + beta_val - 2)
    return model_path


def test_cli_and_mcp_use_same_canonical_math(tmp_path: Path) -> None:
    diff = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,3 +1,6 @@\n"
        "+import os\n"
        "+import subprocess\n"
        " def main():\n"
        "-    print('hello')\n"
        "+    subprocess.run(os.environ.get('CMD', 'echo hi'), shell=True)\n"
    )

    # First pass gives us the target grid cell for this diff.
    prior_result = evaluate_risk(diff, workspace=tmp_path)
    _write_model(tmp_path, prior_result.grid_cell, alpha_val=9, beta_val=3)

    canonical = evaluate_risk(diff, workspace=tmp_path)
    mcp_path = evaluate_action_risk(
        action_type="code_edit",
        description="test",
        files_affected=["src/main.py"],
        diff_text=diff,
        workspace=tmp_path,
    )

    assert round(canonical.p_fail, 6) == round(mcp_path.p_fail, 6)
    assert canonical.decision == mcp_path.decision
    assert canonical.grid_cell == mcp_path.grid_cell


def test_scan_diff_mode_matches_risk_engine(tmp_path: Path) -> None:
    diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,5 @@\n"
        "+import subprocess\n"
        "+import os\n"
        " def run():\n"
        "+    subprocess.run(os.environ['CMD'], shell=True)\n"
    )

    grid = evaluate_risk(diff, workspace=tmp_path).grid_cell
    _write_model(tmp_path, grid, alpha_val=11, beta_val=2)
    expected = evaluate_risk(diff, workspace=tmp_path)

    diff_file = tmp_path / "risk.diff"
    diff_file.write_text(diff)

    runner = CliRunner()
    result = runner.invoke(
        scan_command,
        [str(tmp_path), "--diff", str(diff_file), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert round(payload["p_fail"], 6) == round(expected.p_fail, 6)
    assert payload["decision"] == expected.decision
    assert payload["grid_cell"] == expected.grid_cell
