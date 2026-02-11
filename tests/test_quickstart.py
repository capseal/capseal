from __future__ import annotations

import subprocess
from pathlib import Path

from capseal.cli.cap_format import verify_cap_integrity
from capseal.quickstart import run_quickstart


def _git(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_quickstart_creates_and_verifies_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["git", "init"], repo)
    _git(["git", "config", "user.email", "test@example.com"], repo)
    _git(["git", "config", "user.name", "CapSeal Test"], repo)

    (repo / "main.py").write_text("print('hello')\n")
    _git(["git", "add", "main.py"], repo)
    _git(["git", "commit", "-m", "init"], repo)

    rc = run_quickstart(str(repo), color=False)
    assert rc == 0

    cap_path = repo / ".capseal" / "runs" / "quickstart-demo.cap"
    assert cap_path.exists()
    ok, _msg = verify_cap_integrity(cap_path)
    assert ok
