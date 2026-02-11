"""
CapSeal Operator — Intervention Channel

Writes user commands back to the running agent session via:
  1. MCP intervention file (.capseal/intervention.json)
  2. Direct PTY stdin injection (future, dangerous)

The MCP server reads intervention.json on each gate call and
adjusts behavior accordingly.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


class InterventionChannel:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.intervention_path = workspace / ".capseal" / "intervention.json"

    async def approve_pending(self, target: Optional[str] = None):
        """Override: approve the next gate on target file."""
        await self._write_intervention({
            "action": "approve",
            "target": target,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def deny_pending(self, target: Optional[str] = None):
        """Override: deny the next gate on target file."""
        await self._write_intervention({
            "action": "deny",
            "target": target,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def adjust_threshold(self, new_threshold: float):
        """Adjust the gate threshold for this session."""
        await self._write_intervention({
            "action": "adjust_threshold",
            "value": new_threshold,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def instruct_agent(self, instruction: str):
        """
        Send a natural language instruction to the agent.
        The MCP server injects this into the next tool response.
        """
        await self._write_intervention({
            "action": "instruct",
            "instruction": instruction,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def pause_session(self):
        """Pause the session (MCP server will hold on next gate)."""
        await self._write_intervention({
            "action": "pause",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def resume_session(self):
        """Resume a paused session."""
        await self._write_intervention({
            "action": "resume",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def end_session(self):
        """End the current session."""
        await self._write_intervention({
            "action": "end",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def _write_intervention(self, data: dict):
        """Write intervention command to the intervention file."""
        # Read existing interventions (queue)
        queue = []
        if self.intervention_path.exists():
            try:
                with open(self.intervention_path) as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        queue = existing
                    elif isinstance(existing, dict):
                        queue = [existing]
            except (json.JSONDecodeError, OSError):
                queue = []

        queue.append(data)

        # Write atomically
        tmp_path = self.intervention_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(queue, f, indent=2)
        tmp_path.rename(self.intervention_path)

        print(f"[intervention] Wrote: {data['action']}")
