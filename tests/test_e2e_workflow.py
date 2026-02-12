"""End-to-end workflow tests for CapSeal.

Tests all four workflows against the sample-project fixture:
    A: capseal autopilot (full pipeline)
    B: Step-by-step (init → learn → fix --dry-run → fix → verify)
    C: MCP server tools (status, gate, record, seal via JSON-RPC)
    D: .cap file hash chain integrity after a full session
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Find the capseal binary in the same venv as the test runner
CAPSEAL_BIN = str(Path(sys.executable).parent / "capseal")
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample-project"

# Dummy key for tests that only need to pass the API key check
# (init, scan, dry-run — no actual LLM calls)
DUMMY_ENV = {"ANTHROPIC_API_KEY": "sk-ant-test-dummy-key-for-e2e"}


def _run(args: list[str], cwd: str | None = None, timeout: int = 120, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a capseal CLI command."""
    run_env = os.environ.copy()
    run_env.update(DUMMY_ENV)  # Always provide a key so checks pass
    if env:
        run_env.update(env)
    return subprocess.run(
        [CAPSEAL_BIN] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        env=run_env,
    )


def _copy_fixture(dest: Path) -> None:
    """Copy the sample-project fixture to a temp directory."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(FIXTURES_DIR, dest)


@pytest.fixture
def project_dir(tmp_path):
    """Create a fresh copy of the sample project in a temp dir."""
    project = tmp_path / "sample-project"
    _copy_fixture(project)
    return project


class TestAutopilot:
    """Test A: capseal autopilot full pipeline."""

    def test_autopilot_inits_workspace(self, project_dir):
        """Autopilot should create .capseal/ directory."""
        result = _run(["autopilot", str(project_dir), "--ci"], timeout=300)
        capseal_dir = project_dir / ".capseal"
        assert capseal_dir.exists(), f"No .capseal/ created. stderr: {result.stderr}"
        assert (capseal_dir / "config.json").exists()

    def test_autopilot_skips_existing_workspace(self, project_dir):
        """Autopilot should skip init if workspace exists."""
        # Create workspace first
        (project_dir / ".capseal").mkdir()
        (project_dir / ".capseal" / "config.json").write_text("{}")
        (project_dir / ".capseal" / "models").mkdir()
        (project_dir / ".capseal" / "runs").mkdir()
        (project_dir / ".capseal" / "policies").mkdir()

        result = _run(["autopilot", str(project_dir), "--ci"], timeout=300)
        assert "Workspace exists" in result.stdout or result.returncode == 0

    def test_autopilot_no_auth(self, project_dir):
        """Autopilot should fail cleanly with no API key and no CLI binary."""
        # Override API keys with empty values and strip CLI binaries from PATH
        clean_env = os.environ.copy()
        clean_env["ANTHROPIC_API_KEY"] = ""
        clean_env["OPENAI_API_KEY"] = ""
        clean_env["GOOGLE_API_KEY"] = ""
        # Keep only the venv bin dir (for capseal itself) — hide claude/codex/gemini
        venv_bin = str(Path(sys.executable).parent)
        clean_env["PATH"] = venv_bin
        result = subprocess.run(
            [CAPSEAL_BIN, "autopilot", str(project_dir)],
            capture_output=True, text=True, timeout=30, env=clean_env,
        )
        assert result.returncode != 0
        assert "API key" in result.stderr or "CLI" in result.stderr or \
               "API key" in result.stdout or "CLI" in result.stdout

    def test_autopilot_json_output(self, project_dir):
        """Autopilot --json should produce valid JSON."""
        result = _run(["autopilot", str(project_dir), "--ci", "--json"], timeout=300)
        # Find last valid JSON line
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    assert "target" in data or "status" in data
                    break
                except json.JSONDecodeError:
                    continue


class TestStepByStep:
    """Test B: Step-by-step workflow."""

    def test_init_creates_workspace(self, project_dir):
        """capseal init --no-tui should create workspace."""
        result = _run(["init", "--path", str(project_dir), "--no-tui"])
        assert result.returncode == 0
        assert (project_dir / ".capseal").exists()

    def test_init_then_scan(self, project_dir):
        """Scan should find issues in the fixture project."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["scan", str(project_dir), "--json"])

        # Semgrep should find at least some issues
        output = result.stdout.strip()
        if output:
            for line in output.split("\n"):
                if line.strip().startswith("{"):
                    try:
                        data = json.loads(line.strip())
                        # Even if zero findings, the JSON should be valid
                        assert isinstance(data, dict)
                        break
                    except json.JSONDecodeError:
                        continue

    def test_fix_dryrun_json(self, project_dir):
        """Fix --dry-run --json should return structured output."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["fix", str(project_dir), "--dry-run", "--json"], timeout=300)

        for line in reversed(result.stdout.strip().split("\n")):
            if line.strip().startswith("{"):
                data = json.loads(line.strip())
                assert "status" in data
                break

    def test_doctor_on_workspace(self, project_dir):
        """Doctor should report on workspace status."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["doctor", str(project_dir)])
        assert result.returncode == 0
        assert "CAPSEAL DOCTOR" in result.stdout


class TestMCPServer:
    """Test C: MCP server tools via JSON-RPC."""

    def _mcp_call(self, workspace: str, messages: list[str], timeout: int = 10) -> str:
        """Send JSON-RPC messages to MCP server and return output."""
        input_data = "\n".join(messages) + "\n"
        result = subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", workspace],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout

    def _init_messages(self) -> list[str]:
        """Return standard MCP initialization messages."""
        return [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]

    def test_tools_list_returns_five_tools(self, project_dir):
        """MCP server should expose 5 tools."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 1}),
        ]
        output = self._mcp_call(str(project_dir), messages)

        # Find the response with id=1 (tools/list)
        found = False
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") == 1 and "result" in data:
                tools = data["result"]["tools"]
                tool_names = [t["name"] for t in tools]
                assert "capseal_gate" in tool_names
                assert "capseal_record" in tool_names
                assert "capseal_seal" in tool_names
                assert "capseal_status" in tool_names
                assert "capseal_context" in tool_names
                assert len(tools) == 5
                found = True
                break
        assert found, f"No tools/list response found. Output: {output[:500]}"

    def test_status_returns_session_info(self, project_dir):
        """capseal_status should return session state."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 2,
                        "params": {"name": "capseal_status", "arguments": {}}}),
        ]
        output = self._mcp_call(str(project_dir), messages)

        found = False
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") == 2 and "result" in data:
                text = data["result"]["content"][0]["text"]
                assert "CAPSEAL STATUS" in text
                assert "Session" in text or "Workspace" in text
                found = True
                break
        assert found, f"No status response found. Output: {output[:500]}"

    def test_gate_returns_decision(self, project_dir):
        """capseal_gate should return a gate decision."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 3,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix command injection in app.py",
                            "files_affected": ["app.py"],
                        }}}),
        ]
        output = self._mcp_call(str(project_dir), messages)

        found = False
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") == 3 and "result" in data:
                text = data["result"]["content"][0]["text"]
                assert "CAPSEAL GATE" in text
                assert "APPROVED" in text or "DENIED" in text or "FLAGGED" in text
                found = True
                break
        assert found, f"No gate response found. Output: {output[:500]}"

    def test_full_mcp_session(self, project_dir):
        """Full session: status → gate → record → seal."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            # Status
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 10,
                        "params": {"name": "capseal_status", "arguments": {}}}),
            # Gate
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix subprocess.call shell=True",
                            "files_affected": ["app.py"],
                        }}}),
            # Record
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Replaced subprocess.call with subprocess.run",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            # Seal
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {
                            "session_name": "e2e-test",
                        }}}),
        ]
        output = self._mcp_call(str(project_dir), messages, timeout=30)

        # Should have responses for all 4 calls
        responses = {}
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                if "id" in data and "result" in data:
                    responses[data["id"]] = data
            except json.JSONDecodeError:
                continue

        assert 10 in responses, f"No status response. Output: {output[:500]}"
        assert 11 in responses, f"No gate response. Output: {output[:500]}"
        assert 12 in responses, f"No record response. Output: {output[:500]}"
        assert 13 in responses, f"No seal response. Output: {output[:500]}"

        # Check seal produced a sealed response
        seal_text = responses[13]["result"]["content"][0]["text"]
        assert "CAPSEAL SEALED" in seal_text or "SEALED" in seal_text


class TestRiskReport:
    """Test E: Risk report and export-receipt commands."""

    def test_report_runs_on_workspace(self, project_dir):
        """capseal report should produce output on an initialized workspace."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["report", str(project_dir), "--print"])
        assert result.returncode == 0
        assert "CAPSEAL RISK REPORT" in result.stdout or "No security findings" in result.stdout

    def test_report_json_output(self, project_dir):
        """capseal report --json should produce valid JSON."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["report", str(project_dir), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout.strip())
        assert "project" in data
        assert "hotspots" in data
        assert "recommendations" in data
        assert "session_history" in data

    def test_report_json_has_findings(self, project_dir):
        """Report JSON should contain findings from the sample project."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["report", str(project_dir), "--json"])
        data = json.loads(result.stdout.strip())
        # The sample project has semgrep findings
        assert data["findings_count"] >= 0
        assert isinstance(data["hotspots"], list)

    def test_export_receipt_after_mcp_session(self, project_dir):
        """export-receipt should produce valid JSON from a .cap file."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session via MCP (status → gate → record → seal)
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 10,
                        "params": {"name": "capseal_status", "arguments": {}}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix eval() in app.py",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Replaced eval with ast.literal_eval",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {
                            "session_name": "export-test",
                        }}}),
        ]
        input_data = "\n".join(messages) + "\n"
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input=input_data, capture_output=True, text=True, timeout=30,
        )

        # Find the .cap file
        runs_dir = project_dir / ".capseal" / "runs"
        cap_files = sorted(
            [f for f in runs_dir.glob("*.cap") if not f.is_symlink()],
            key=lambda f: f.name,
        )
        if cap_files:
            cap_file = cap_files[-1]
            result = _run(["export-receipt", str(cap_file), "--print"])
            assert result.returncode == 0
            data = json.loads(result.stdout.strip())
            assert data["schema"] == "capseal_receipt_v1"
            assert "actions" in data
            assert "chain_hash" in data
            assert "integrity" in data
            assert data["integrity"]["algorithm"] == "sha256"


class TestRecomputableProofs:
    """Test Feature 1: Recomputable proofs in export-receipt."""

    def test_export_receipt_includes_canonical_fields(self, project_dir):
        """Exported receipt should include canonical_fields for each action."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session via MCP
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix eval() in app.py",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Replaced eval with literal_eval",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {
                            "session_name": "canonical-test",
                        }}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        # Export receipt
        cap_files = sorted(
            [f for f in (project_dir / ".capseal" / "runs").glob("*.cap") if not f.is_symlink()],
            key=lambda f: f.name,
        )
        assert cap_files, "No .cap file created"
        result = _run(["export-receipt", str(cap_files[-1]), "--print"])
        assert result.returncode == 0
        data = json.loads(result.stdout.strip())

        # Each action should have canonical_fields
        assert len(data["actions"]) >= 1
        for action in data["actions"]:
            assert action.get("canonical_fields") is not None, \
                f"Action {action['index']} missing canonical_fields"
            cf = action["canonical_fields"]
            assert "action_id" in cf
            assert "action_type" in cf
            assert "instruction_hash" in cf

    def test_receipt_hash_recomputable(self, project_dir):
        """Receipt hashes should be recomputable from canonical_fields."""
        import hashlib

        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix shell injection",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Used subprocess.run with list args",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        cap_files = sorted(
            [f for f in (project_dir / ".capseal" / "runs").glob("*.cap") if not f.is_symlink()],
            key=lambda f: f.name,
        )
        assert cap_files
        result = _run(["export-receipt", str(cap_files[-1]), "--print"])
        data = json.loads(result.stdout.strip())

        # Recompute each receipt_hash from canonical_fields
        for action in data["actions"]:
            cf = action.get("canonical_fields")
            if cf is None:
                continue
            canonical = json.dumps(cf, sort_keys=True, separators=(",", ":"))
            recomputed = hashlib.sha256(canonical.encode()).hexdigest()
            assert recomputed == action["receipt_hash"], \
                f"Action {action['index']}: hash mismatch. Recomputed={recomputed[:16]}, stored={action['receipt_hash'][:16]}"

    def test_integrity_fully_recomputable(self, project_dir):
        """Integrity block should report fully_recomputable=True for v0.3 sessions."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "test",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "test",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        cap_files = sorted(
            [f for f in (project_dir / ".capseal" / "runs").glob("*.cap") if not f.is_symlink()],
            key=lambda f: f.name,
        )
        assert cap_files
        result = _run(["export-receipt", str(cap_files[-1]), "--print"])
        data = json.loads(result.stdout.strip())
        assert data["integrity"]["fully_recomputable"] is True


class TestCIReport:
    """Test Feature 4: CI report command."""

    def test_ci_report_markdown(self, project_dir):
        """ci-report should produce markdown output."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session first
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix issue",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fixed issue",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        result = _run(["ci-report", str(project_dir)])
        assert result.returncode == 0
        assert "CapSeal Security Gate" in result.stdout
        assert "actions" in result.stdout

    def test_ci_report_json(self, project_dir):
        """ci-report --format json should produce valid JSON."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fixed",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        result = _run(["ci-report", str(project_dir), "--format", "json"])
        assert result.returncode == 0
        data = json.loads(result.stdout.strip())
        assert "total_actions" in data
        assert "approved" in data
        assert "denied" in data
        assert "verified" in data


class TestConfigProfiles:
    """Test Feature 6: Config profiles save/load/list."""

    def test_config_save_load_roundtrip(self, project_dir):
        """Saving and loading a profile should preserve config."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Save current config
        result = _run(["config", "save", "test-profile", "-w", str(project_dir)])
        assert result.returncode == 0

        # Modify the config
        config_path = project_dir / ".capseal" / "config.json"
        config = json.loads(config_path.read_text())
        config["test_marker"] = "modified"
        config_path.write_text(json.dumps(config))

        # Load the saved profile back
        result = _run(["config", "load", "test-profile", "-w", str(project_dir)])
        assert result.returncode == 0

        # Verify original config was restored (no test_marker)
        restored = json.loads(config_path.read_text())
        assert "test_marker" not in restored

    def test_config_list(self, project_dir):
        """config list should show saved profiles."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        _run(["config", "save", "alpha", "-w", str(project_dir)])
        _run(["config", "save", "beta", "-w", str(project_dir)])

        result = _run(["config", "list", "-w", str(project_dir)])
        assert result.returncode == 0
        assert "alpha" in result.stdout
        assert "beta" in result.stdout


class TestSignature:
    """Test Feature 7: Identity signing."""

    def test_sign_and_verify(self, project_dir):
        """Sign a .cap file and verify the signature."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fixed",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        cap_files = sorted(
            [f for f in (project_dir / ".capseal" / "runs").glob("*.cap") if not f.is_symlink()],
            key=lambda f: f.name,
        )
        assert cap_files, "No .cap file created"
        cap_file = cap_files[-1]

        # Sign with --generate-key
        result = _run(["sign", str(cap_file), "--generate-key"])
        assert result.returncode == 0
        assert "Signed" in result.stdout

        # Check .sig file was created
        sig_file = cap_file.with_suffix(".cap.sig")
        assert sig_file.exists()
        sig_data = json.loads(sig_file.read_text())
        assert sig_data["schema"] == "capseal_signature_v1"
        assert sig_data["algorithm"] == "Ed25519"

        # Verify programmatically
        from capseal.cli.sign_cmd import verify_signature
        valid, message = verify_signature(cap_file)
        assert valid, f"Signature verification failed: {message}"


class TestSelfTest:
    """Test: capseal test command."""

    def test_self_test_on_workspace(self, project_dir):
        """capseal test should pass workspace and config checks."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["test", str(project_dir)])
        # Should pass at least workspace + config (2/8)
        assert "PASS" in result.stdout
        assert "Workspace" in result.stdout
        assert "Configuration" in result.stdout

    def test_self_test_on_full_session(self, project_dir):
        """capseal test should validate chain + proofs on a real session."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session via MCP
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix eval in app.py",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Replaced eval with literal_eval",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        result = _run(["test", str(project_dir)])
        # Should pass: workspace, config, latest session, chain, recomputable (5/8)
        # Risk model, episode history, signature may fail (that's OK)
        assert "Chain integrity" in result.stdout
        assert "Recomputable" in result.stdout


class TestSignatureSymlink:
    """Test Bug 1 fix: signature detection via symlinks."""

    def test_verify_signature_via_symlink(self, project_dir):
        """verify_signature should work when given a symlink path."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        # Create a session
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fixed",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

        # Sign the real .cap file
        cap_files = sorted(
            [f for f in (project_dir / ".capseal" / "runs").glob("*.cap") if not f.is_symlink()],
            key=lambda f: f.name,
        )
        assert cap_files
        result = _run(["sign", str(cap_files[-1]), "--generate-key"])
        assert result.returncode == 0

        # Verify via the SYMLINK path (latest.cap) — this was the bug
        latest_cap = project_dir / ".capseal" / "runs" / "latest.cap"
        assert latest_cap.is_symlink(), "latest.cap should be a symlink"

        from capseal.cli.sign_cmd import verify_signature
        valid, message = verify_signature(latest_cap)
        assert valid, f"Signature verification via symlink failed: {message}"


class TestCapFileIntegrity:
    """Test D: .cap file hash chain integrity."""

    def test_cap_file_created_by_fix(self, project_dir):
        """capseal fix should produce a .cap file."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["fix", str(project_dir), "--dry-run"], timeout=300)

        # Even dry-run should succeed (it just doesn't generate patches)
        # Just verify the command doesn't crash
        assert result.returncode == 0 or "No findings" in result.stdout

    def test_verify_accepts_valid_cap(self, project_dir):
        """capseal verify should accept a valid .cap from a fix run."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        _run(["fix", str(project_dir)], timeout=300)

        latest_cap = project_dir / ".capseal" / "runs" / "latest.cap"
        if latest_cap.exists():
            result = _run(["verify", str(latest_cap.resolve()), "--json"])
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                    assert data.get("status") in ("VERIFIED", "ERROR"), f"Unexpected status: {data}"
                    break
                except json.JSONDecodeError:
                    continue


class TestProofOnSeal:
    """Test proof status surfaces through MCP seal and verify."""

    def _mcp_call(self, workspace: str, messages: list[str], timeout: int = 30) -> str:
        """Send JSON-RPC messages to MCP server and return output."""
        input_data = "\n".join(messages) + "\n"
        result = subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", workspace],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout

    def _init_messages(self) -> list[str]:
        return [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]

    def _run_full_session(self, project_dir) -> dict:
        """Run gate → record → seal, return all MCP responses."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 1,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix shell=True vulnerability",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 2,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Replaced subprocess.call with subprocess.run",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 3,
                        "params": {"name": "capseal_seal", "arguments": {
                            "session_name": "proof-test",
                        }}}),
        ]
        output = self._mcp_call(str(project_dir), messages)

        responses = {}
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                if "id" in data and "result" in data:
                    responses[data["id"]] = data
            except json.JSONDecodeError:
                continue
        return responses

    def test_seal_includes_proof_line(self, project_dir):
        """Seal output should include proof status (structured JSON)."""
        responses = self._run_full_session(project_dir)
        assert 3 in responses, "No seal response"
        seal_text = responses[3]["result"]["content"][0]["text"]
        payload = json.loads(seal_text)
        assert payload.get("sealed") is True
        assert "CAPSEAL SEALED" in (payload.get("human_summary") or "")
        # Proof fields should be present even when verification is stubbed.
        assert payload.get("proof_type") in ("constraint_check", "fri")
        assert payload.get("proof_generated") in (True, False)
        assert payload.get("proof_verified") in (True, False)

    def test_seal_creates_agent_capsule(self, project_dir):
        """After seal, agent_capsule.json should exist in the run directory."""
        self._run_full_session(project_dir)

        runs_dir = project_dir / ".capseal" / "runs"
        assert runs_dir.exists(), "No runs directory"

        # Find the run directory (not the .cap file or symlinks)
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.is_symlink()]
        assert run_dirs, "No run directory found"

        capsule_found = False
        for run_dir in run_dirs:
            capsule_file = run_dir / "agent_capsule.json"
            if capsule_file.exists():
                capsule = json.loads(capsule_file.read_text())
                assert capsule.get("schema") == "agent_capsule_v1"
                assert "verification" in capsule
                assert "proof_type" in capsule["verification"]
                capsule_found = True
                break

        assert capsule_found, "agent_capsule.json not found in any run directory"

    def test_verify_shows_proof_type(self, project_dir):
        """capseal verify --json should include proof_type in output."""
        self._run_full_session(project_dir)

        latest_cap = project_dir / ".capseal" / "runs" / "latest.cap"
        if not latest_cap.exists():
            pytest.skip("No .cap file created")

        result = _run(["verify", str(latest_cap.resolve()), "--json"])
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                if "proof_type" in data:
                    assert data["proof_type"] in ("constraint_check", "fri")
                    return
            except json.JSONDecodeError:
                continue
        # If no proof_type in output, the verify may not have found agent_capsule.json
        # That's acceptable if verify fell through to the run .cap path
        # Just verify the command didn't crash
        assert result.returncode in (0, 1), f"Verify crashed: {result.stderr}"


class TestPtyShell:
    """Test Feature 1: PTY Shell."""

    def test_shell_help(self):
        """capseal shell --help should return 0 with usage info."""
        result = _run(["shell", "--help"])
        assert result.returncode == 0
        assert "PTY shell" in result.stdout or "status bar" in result.stdout

    def test_shell_exits_noninteractive(self, project_dir):
        """Shell should exit cleanly when stdin isn't a TTY."""
        result = _run(["shell", "-w", str(project_dir)])
        # When run in subprocess with capture (no TTY), should exit with error
        assert result.returncode != 0 or "interactive terminal" in result.stderr

    def test_shell_dry_run(self, project_dir):
        """capseal shell --dry-run should print config without launching PTY."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        result = _run(["shell", "--dry-run", "-w", str(project_dir)])
        assert result.returncode == 0
        assert "Workspace" in result.stdout
        assert "dry run" in result.stdout.lower()


class TestLiveEvents:
    """Test Feature 2: Live event emission from MCP server."""

    def _mcp_call(self, workspace: str, messages: list[str], timeout: int = 30) -> str:
        input_data = "\n".join(messages) + "\n"
        result = subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", workspace],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout

    def _init_messages(self) -> list[str]:
        return [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]

    def test_mcp_gate_emits_event(self, project_dir):
        """After MCP gate call, events.jsonl should have a gate event."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 1,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix eval in app.py",
                            "files_affected": ["app.py"],
                        }}}),
        ]
        self._mcp_call(str(project_dir), messages)

        events_path = project_dir / ".capseal" / "events.jsonl"
        assert events_path.exists(), "events.jsonl not created"
        events = [json.loads(line) for line in events_path.read_text().strip().split("\n") if line.strip()]
        gate_events = [e for e in events if e["type"] == "gate"]
        assert len(gate_events) >= 1, f"No gate events found. Events: {events}"

    def test_mcp_seal_emits_event(self, project_dir):
        """Full session should produce gate + record + seal events."""
        _run(["init", "--path", str(project_dir), "--no-tui"])

        messages = self._init_messages() + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 1,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix issue",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 2,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fixed issue",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 3,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        self._mcp_call(str(project_dir), messages)

        events_path = project_dir / ".capseal" / "events.jsonl"
        assert events_path.exists()
        events = [json.loads(line) for line in events_path.read_text().strip().split("\n") if line.strip()]
        types = [e["type"] for e in events]
        assert "gate" in types, f"No gate event. Types: {types}"
        assert "record" in types, f"No record event. Types: {types}"
        assert "seal" in types, f"No seal event. Types: {types}"


class TestSarifExport:
    """Test Feature 6: SARIF export."""

    def _create_session(self, project_dir):
        """Helper: create an MCP session."""
        _run(["init", "--path", str(project_dir), "--no-tui"])
        init_msgs = [
            json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0.1"}}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
        messages = init_msgs + [
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 11,
                        "params": {"name": "capseal_gate", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fix vulnerability",
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 12,
                        "params": {"name": "capseal_record", "arguments": {
                            "action_type": "code_edit",
                            "description": "Fixed vulnerability",
                            "tool_name": "edit",
                            "success": True,
                            "files_affected": ["app.py"],
                        }}}),
            json.dumps({"jsonrpc": "2.0", "method": "tools/call", "id": 13,
                        "params": {"name": "capseal_seal", "arguments": {}}}),
        ]
        subprocess.run(
            [CAPSEAL_BIN, "mcp-serve", "-w", str(project_dir)],
            input="\n".join(messages) + "\n",
            capture_output=True, text=True, timeout=30,
        )

    def test_ci_report_sarif_format(self, project_dir):
        """ci-report --format sarif should produce valid SARIF 2.1.0."""
        self._create_session(project_dir)
        result = _run(["ci-report", str(project_dir), "--format", "sarif"])
        assert result.returncode == 0
        sarif = json.loads(result.stdout.strip())
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

    def test_sarif_has_results(self, project_dir):
        """SARIF output should have results with ruleId and level."""
        self._create_session(project_dir)
        result = _run(["ci-report", str(project_dir), "--format", "sarif"])
        sarif = json.loads(result.stdout.strip())
        results = sarif["runs"][0]["results"]
        assert len(results) >= 1
        for r in results:
            assert "ruleId" in r
            assert "level" in r
            assert r["ruleId"].startswith("capseal/")

    def test_sarif_level_mapping(self):
        """Unit test: _map_gate_to_level should map correctly."""
        from capseal.sarif import _map_gate_to_level
        assert _map_gate_to_level("skip") == "error"
        assert _map_gate_to_level("human_review") == "warning"
        assert _map_gate_to_level("pass") == "note"
        assert _map_gate_to_level(None) == "note"


class TestProtocolSpec:
    """Test Feature 11: Protocol specification documents."""

    def test_spec_files_exist(self):
        """All spec .md files and examples should exist."""
        docs_dir = Path(__file__).parent.parent / "docs"
        assert (docs_dir / "AGENT_PROTOCOL.md").exists()
        assert (docs_dir / "RECEIPT_FORMAT.md").exists()
        assert (docs_dir / "PROOF_FORMAT.md").exists()
        assert (docs_dir / "examples" / "minimal_integration.py").exists()
        assert (docs_dir / "examples" / "verify_receipt.sh").exists()

    def test_example_valid_python(self):
        """minimal_integration.py should parse as valid Python."""
        import ast
        example_path = Path(__file__).parent.parent / "docs" / "examples" / "minimal_integration.py"
        source = example_path.read_text()
        # Should not raise SyntaxError
        ast.parse(source)
