"""
CapSeal Operator — Message Composer

Generates human-readable messages at three detail levels:
  short_text  — 1-2 lines for text notification
  full_text   — Paragraph for detailed text message
  voice_text  — Natural speech for PersonaPlex synthesis

The composer turns raw event data into something a human
can glance at on their phone and immediately understand.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional
import html as _html


def _esc(text: str) -> str:
    """HTML-escape text for safe Telegram rendering."""
    return _html.escape(str(text)) if text else ""


def _callback_target(filepath: str) -> str:
    """Encode filepath for Telegram callback_data (max 64 bytes)."""
    if len(f"approve:{filepath}") > 64:
        return PurePosixPath(filepath).name
    return filepath


@dataclass
class Message:
    short_text: str
    full_text: str
    voice_text: str
    buttons: Optional[list] = None  # For Telegram inline keyboards


class MessageComposer:
    """Composes human-readable messages from CapSeal events."""

    def compose(self, event: dict, context, score: float) -> Message:
        etype = event.get("type", "")
        data = event.get("data", {})

        if etype == "gate":
            return self._compose_gate(data, context, score)
        elif etype == "session_start":
            return self._compose_session_start(data, context)
        elif etype == "session_seal":
            return self._compose_session_seal(data, context)
        elif etype == "chain_break":
            return self._compose_chain_break(data, context)
        elif etype == "budget_alert":
            return self._compose_budget_alert(data, context)
        elif etype == "agent_waiting":
            return self._compose_agent_waiting(data, context)
        elif etype == "agent_approval_request":
            return self._compose_approval_request(data, context)
        else:
            return Message(
                short_text=f"📋 Event: {etype}",
                full_text=f"📋 CapSeal event: {etype}\nData: {data}",
                voice_text=f"Got an event of type {etype}.",
            )

    def _compose_gate(self, data: dict, context, score: float) -> Message:
        raw_decision = str(data.get("decision", "unknown")).strip().lower()
        if raw_decision in {"deny", "denied", "skip"}:
            decision = "denied"
        elif raw_decision in {"approve", "approved", "pass"}:
            decision = "approved"
        elif raw_decision in {"flag", "flagged", "human_review"}:
            decision = "flagged"
        else:
            decision = raw_decision
        files = data.get("files", [])
        p_fail = data.get("p_fail")
        label = (data.get("label") or "").strip()
        diff = data.get("diff", "")
        risk_factors = data.get("risk_factors", "")
        action_type = data.get("action_type", "edit")

        # File display
        file_str = ", ".join(files) if files else "unknown file"
        file_short = files[0] if files else "unknown"
        if len(files) > 1:
            file_short = f"{files[0]} +{len(files)-1}"

        # p_fail display
        p_str = f" (p_fail={p_fail:.2f})" if p_fail is not None else ""
        p_pct = f"{int(p_fail * 100)}%" if p_fail is not None else "unknown"
        p_voice = f"Risk score is {int(p_fail * 100)} percent" if p_fail is not None else "Risk level unknown"

        if decision == "denied":
            return self._compose_denial(
                file_short, file_str, p_str, p_pct, p_voice,
                diff, risk_factors, label, data, context, score
            )
        elif decision == "flagged":
            return self._compose_flagged(
                file_short, file_str, p_str, p_pct, p_voice,
                diff, risk_factors, label, data, context, score
            )
        elif decision == "approved" and p_fail is not None and p_fail >= 0.5:
            return self._compose_risky_approval(
                file_short, file_str, p_str, p_pct, p_voice,
                diff, risk_factors, label, data, context, score
            )
        else:
            return self._compose_routine_approval(
                file_short, file_str, p_str, p_pct, p_voice,
                label, data, context, score
            )

    def _gate_buttons(self, files: list[str]) -> list[dict]:
        target = _callback_target(files[0]) if files else ""
        return [
            {"text": "✅ allow once", "callback": f"approve:{target}"},
            {"text": "🛑 block once", "callback": f"deny:{target}"},
            {"text": "⏸ pause session", "callback": "pause:"},
            {"text": "🔍 show why", "callback": f"investigate:{target}"},
            {"text": "📦 seal now", "callback": "end:"},
        ]

    def _compose_denial(self, file_short, file_str, p_str, p_pct,
                        p_voice, diff, risk_factors, label, data, context, score):
        # Check for consecutive denial streak
        files = data.get("files", [])
        streak_info = ""
        streak_voice = ""
        for f in files:
            streak = context.consecutive_denials.get(f, 0)
            if streak >= 2:
                streak_info = f"\n⚠️ {streak} consecutive denials on {f}!"
                streak_voice = f"That's {streak} denials in a row on this file. "

        # Diff preview
        diff_preview = ""
        if diff:
            lines = diff.strip().split("\n")[:6]
            diff_preview = "\n<pre>" + _esc("\n".join(lines)) + "</pre>"

        # Risk factors
        risk_line = ""
        risk_voice = ""
        if risk_factors:
            risk_line = f"\n📊 Risk: {risk_factors}"
            risk_voice = f"The risk is: {risk_factors}. "

        headline = label or file_short
        short = f"🛑 Blocked for safety — {headline} (chance this breaks: {p_pct})"

        full = (
            f"🛑 <b>Blocked for safety</b>\n"
            f"Change: {_esc(headline)}\n"
            f"Chance this breaks: {p_pct}"
            f"{streak_info}"
            f"{risk_line}"
            f"{diff_preview}\n\n"
            f"Use the buttons below: <b>allow once</b>, <b>block once</b>, <b>pause session</b>, or <b>seal now</b>."
        )

        voice = (
            f"I blocked this change for safety on {file_short}. "
            f"{p_voice}. "
            f"{streak_voice}"
            f"{risk_voice}"
            f"Say allow once, block once, pause session, or seal now."
        )

        buttons = self._gate_buttons(files)

        return Message(short_text=short, full_text=full,
                       voice_text=voice, buttons=buttons)

    def _compose_risky_approval(self, file_short, file_str, p_str, p_pct,
                                p_voice, diff, risk_factors, label, data, context, score):
        diff_preview = ""
        if diff:
            lines = diff.strip().split("\n")[:4]
            diff_preview = "\n<pre>" + _esc("\n".join(lines)) + "</pre>"

        risk_line = ""
        risk_voice = ""
        if risk_factors:
            risk_line = f"\n📊 Risk: {risk_factors}"
            risk_voice = f"{risk_factors}. "

        headline = label or file_short
        short = f"🟡 Proceed with caution — {headline} (chance this breaks: {p_pct})"

        full = (
            f"🟡 <b>Proceed with caution</b>\n"
            f"Change: {_esc(headline)}\n"
            f"Chance this breaks: {p_pct} (higher than normal)"
            f"{risk_line}"
            f"{diff_preview}\n\n"
            f"Use <b>show why</b> for details, or choose <b>block once</b>."
        )

        voice = (
            f"This was allowed, but with caution, for {file_short}. "
            f"{p_voice}. "
            f"{risk_voice}"
            f"Say block once if you want me to stop it next time."
        )

        return Message(short_text=short, full_text=full, voice_text=voice, buttons=self._gate_buttons(data.get("files", [])))

    def _compose_flagged(self, file_short, file_str, p_str, p_pct,
                         p_voice, diff, risk_factors, label, data, context, score):
        diff_preview = ""
        if diff:
            lines = diff.strip().split("\n")[:4]
            diff_preview = "\n<pre>" + _esc("\n".join(lines)) + "</pre>"

        risk_line = f"\n📊 Risk: {risk_factors}" if risk_factors else ""
        headline = label or file_short

        short = f"🟡 Proceed with caution — {headline} (chance this breaks: {p_pct})"
        full = (
            f"🟡 <b>Proceed with caution</b>\n"
            f"Change: {_esc(headline)}\n"
            f"Chance this breaks: {p_pct}"
            f"{risk_line}"
            f"{diff_preview}\n\n"
            f"Use <b>show why</b> for details, then choose <b>allow once</b> or <b>block once</b>."
        )
        voice = (
            f"This change needs caution for {file_short}. "
            f"{p_voice}. "
            f"Say allow once or block once."
        )
        return Message(short_text=short, full_text=full, voice_text=voice, buttons=self._gate_buttons(data.get("files", [])))

    def _compose_routine_approval(self, file_short, file_str, p_str, p_pct,
                                  p_voice, label, data, context, score):
        headline = label or file_short
        short = f"✅ Safe to proceed — {headline} (chance this breaks: {p_pct})"
        full = (
            f"✅ <b>Safe to proceed</b>\n"
            f"Change: {_esc(headline)}\n"
            f"Chance this breaks: {p_pct}\n\n"
            f"If you want details, tap <b>show why</b>."
        )
        voice = f"This change looks safe to proceed for {file_short}. Low risk."

        return Message(short_text=short, full_text=full, voice_text=voice, buttons=self._gate_buttons(data.get("files", [])))

    def _compose_session_start(self, data, context):
        agent = data.get("agent", "agent")
        workspace = data.get("workspace", "")
        model = data.get("model", "")

        short = f"🚀 Session started: {agent}"

        full = (
            f"🚀 <b>CapSeal Session Started</b>\n"
            f"Agent: {_esc(agent)}\n"
            f"Workspace: {_esc(workspace)}\n"
            f"Model: {_esc(model)}\n\n"
            f"Monitoring active. I'll notify you of important events."
        )

        voice = (
            f"New session started with {agent}. "
            f"I'm monitoring. I'll stay quiet unless something interesting happens."
        )

        return Message(short_text=short, full_text=full, voice_text=voice)

    def _compose_session_seal(self, data, context):
        total = context.total_actions
        denied = context.denials
        approved = context.approvals
        chain = "intact" if context.chain_intact else "BROKEN"
        duration = context.duration_str
        trust = context.trust_score

        chain_emoji = "✅" if context.chain_intact else "❌"

        short = f"📦 Session sealed: {total} actions, {denied} denied, chain {chain}"

        full = (
            f"📦 <b>Session Sealed</b>\n"
            f"Actions: {total} total ({approved}✓ {denied}✗)\n"
            f"Trust: {trust:.0%}\n"
            f"Chain: {chain_emoji} {chain}\n"
            f"Duration: {duration}\n\n"
            f"Receipt is sealed and verifiable."
        )

        voice = (
            f"Session's done. {total} actions total, {denied} blocked. "
            f"Chain is {chain} and the receipt is sealed. "
            + ("Nice clean run." if denied == 0 else
               f"{denied} actions were blocked. Overall trust score: {int(trust * 100)} percent.")
        )

        return Message(short_text=short, full_text=full, voice_text=voice)

    def _compose_chain_break(self, data, context):
        action_num = data.get("action_number", "?")
        file_affected = data.get("file", "unknown")

        short = f"🚨 CHAIN BREAK DETECTED at action {action_num}"

        full = (
            f"🚨 <b>CHAIN BREAK DETECTED</b>\n\n"
            f"Action #{action_num} on {_esc(file_affected)}\n"
            f"The receipt chain has been broken — previous hash doesn't match.\n"
            f"This could indicate tampering or a corrupted write.\n\n"
            f"<b>Immediate action recommended.</b>"
        )

        voice = (
            f"Critical alert — the receipt chain just broke at action {action_num}. "
            f"This means someone or something tampered with the action sequence. "
            f"You should investigate immediately."
        )

        buttons = [
            {"text": "▶️ Resume", "callback": "resume:"},
            {"text": "⏹ End Session", "callback": "end:"},
            {"text": "🔍 Investigate", "callback": "investigate:"},
        ]

        return Message(short_text=short, full_text=full,
                       voice_text=voice, buttons=buttons)

    def _compose_budget_alert(self, data, context):
        spent = data.get("spent", 0)
        cap = data.get("cap", 0)
        pct = data.get("percent", 0)

        short = f"💰 Budget alert: ${spent:.2f} / ${cap:.2f} ({pct}%)"

        full = (
            f"💰 <b>Budget Alert</b>\n"
            f"Spent: ${spent:.2f} of ${cap:.2f} ({pct}% used)\n"
            f"Consider wrapping up the session."
        )

        voice = (
            f"Budget heads up — you've spent {pct} percent of your session budget. "
            f"That's {spent:.2f} dollars out of {cap:.2f}."
        )

        return Message(short_text=short, full_text=full, voice_text=voice)

    def _compose_agent_waiting(self, data, context):
        prompt = data.get("prompt_text", "")
        if len(prompt) > 150:
            prompt = prompt[-150:]

        short = f"💬 Agent is waiting for input"

        full = (
            f"💬 <b>Agent Waiting</b>\n"
            f"The agent appears to be waiting for your response:\n"
            f"<pre>{_esc(prompt)}</pre>"
        )

        voice = "The agent is waiting for your input. Check the terminal."

        return Message(short_text=short, full_text=full, voice_text=voice)

    def _compose_approval_request(self, data, context):
        prompt = data.get("prompt_text", "")
        if len(prompt) > 200:
            prompt = prompt[-200:]
        matched = data.get("matched_pattern", "accept edits")

        short = f"🔔 Agent wants approval: {matched}"

        full = (
            f"🔔 <b>Agent Approval Request</b>\n"
            f"The agent is asking to apply changes:\n"
            f"<pre>{_esc(prompt[-150:])}</pre>\n\n"
            f"Accept or reject?"
        )

        voice = (
            "Hey, the agent is asking you to accept some edits. "
            "Tap accept to let it through, or reject to deny."
        )

        buttons = [
            {"text": "\u2705 Accept", "callback": "pty_accept:"},
            {"text": "\u274c Reject", "callback": "pty_reject:"},
        ]

        return Message(short_text=short, full_text=full,
                       voice_text=voice, buttons=buttons)
