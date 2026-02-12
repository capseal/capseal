#!/usr/bin/env python3
"""
CapSeal Operator Agent — Main Daemon

Watches events.jsonl from CapSeal sessions, scores significance,
composes human-readable messages, and dispatches notifications
through configured channels (Telegram, WhatsApp, iMessage, voice).

Usage:
    capseal operator /path/to/project
    capseal operator . --config ~/.capseal/operator.json
"""

import asyncio
import contextlib
import errno
import json
import signal
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from .significance import SignificanceFilter
from .composer import MessageComposer, Message
from .config import load_config, DEFAULT_CONFIG
from .intervention import InterventionChannel


@dataclass
class SessionContext:
    """Tracks state across a CapSeal session for contextual notifications."""
    session_id: Optional[str] = None
    started_at: Optional[datetime] = None
    total_actions: int = 0
    approvals: int = 0
    denials: int = 0
    files_touched: dict = field(default_factory=dict)  # filename -> count
    consecutive_denials: dict = field(default_factory=dict)  # filename -> streak
    last_event_type: Optional[str] = None
    chain_intact: bool = True
    sealed: bool = False

    def update(self, event: dict):
        data = event.get("data", {})
        etype = event.get("type", "")

        if etype == "session_start":
            self.session_id = data.get("session_id")
            self.started_at = datetime.now(timezone.utc)
            self.total_actions = 0
            self.approvals = 0
            self.denials = 0
            self.files_touched = {}
            self.consecutive_denials = {}
            self.chain_intact = True
            self.sealed = False

        elif etype == "gate":
            self.total_actions += 1
            decision = data.get("decision", "")
            files = data.get("files", [])

            if decision == "approved":
                self.approvals += 1
                for f in files:
                    self.files_touched[f] = self.files_touched.get(f, 0) + 1
                    self.consecutive_denials[f] = 0
            elif decision == "denied":
                self.denials += 1
                for f in files:
                    self.files_touched[f] = self.files_touched.get(f, 0) + 1
                    self.consecutive_denials[f] = self.consecutive_denials.get(f, 0) + 1

        elif etype == "chain_break":
            self.chain_intact = False

        elif etype == "session_seal":
            self.sealed = True

        self.last_event_type = etype

    @property
    def trust_score(self) -> float:
        if self.total_actions == 0:
            return 1.0
        return self.approvals / self.total_actions

    @property
    def duration_str(self) -> str:
        if not self.started_at:
            return "0s"
        delta = datetime.now(timezone.utc) - self.started_at
        mins = int(delta.total_seconds() // 60)
        secs = int(delta.total_seconds() % 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def summary(self) -> str:
        return (
            f"Session {self.session_id or 'active'}: "
            f"{self.total_actions} actions ({self.approvals}\u2713 {self.denials}\u2717), "
            f"trust {self.trust_score:.0%}, "
            f"chain {'intact' if self.chain_intact else 'BROKEN'}, "
            f"duration {self.duration_str}"
        )


class OperatorDaemon:
    def __init__(self, workspace: Path, config: dict):
        self.workspace = workspace
        self._instance_lock_fp = None
        self._acquire_instance_lock()
        self.events_path = workspace / ".capseal" / "events.jsonl"
        self.pty_events_path = workspace / ".capseal" / "pty_events.jsonl"
        self.home_pty_events_path = Path.home() / ".capseal" / "pty_events.jsonl"
        self.status_path = workspace / ".capseal" / "operator_status.json"
        # Also write a global status file so any TUI can detect a running operator in another repo.
        self.home_status_path = Path.home() / ".capseal" / "operator_status.json"
        self.voice_control_path = workspace / ".capseal" / "voice_control.json"
        self.config = config
        self.context = SessionContext()
        self.significance = SignificanceFilter(config)
        self.composer = MessageComposer()
        self.channels = []
        self.last_file_pos = 0
        self.last_file_size = 0
        self.last_pty_file_pos = 0
        self.last_pty_file_size = 0
        self.last_home_pty_file_pos = 0
        self.last_home_pty_file_size = 0
        self.replay = False
        self.running = False
        self._last_status_write = 0.0
        self._last_alert_ts = None
        self._events_processed = 0
        self._seen_timestamps = set()
        self.last_event_time = time.time()
        self._voice_active = False
        self._voice_speak_lock = asyncio.Lock()

        self.intervention = InterventionChannel(workspace)
        self._init_channels(config)

        # Voice synthesis (Phase 5)
        self.voice = None
        voice_cfg = config.get("voice", {})
        if voice_cfg.get("enabled"):
            try:
                from .voice import VoiceSynthesizer
                self.voice = VoiceSynthesizer(voice_cfg)
                print(f"[operator] Voice synthesis enabled ({voice_cfg.get('provider', 'openai')})")
            except Exception as e:
                print(f"[operator] Voice init failed: {e}")

        # Live voice call (Phase 6)
        self.voice_call = None
        self._voice_listen_task = None
        self._voice_maintenance_task = None
        self._voice_resume_task = None
        self._voice_listen_commands = False
        self._voice_pod_stopped = False
        self._last_voice_reconnect_attempt = 0.0
        self._voice_reconnect_interval_seconds = 30
        self._voice_idle_seconds = 1800
        self._voice_resume_wait_seconds = 45
        self._runpod_pod_id = None
        self._runpod_api_key = None
        if voice_cfg.get("live_call"):
            try:
                from .voice_call import VoiceCallManager
                self.voice_call = VoiceCallManager(voice_cfg)
                self._voice_listen_commands = bool(voice_cfg.get("listen_commands", False))
                self._voice_reconnect_interval_seconds = int(voice_cfg.get("reconnect_interval_seconds", 30))
                self._voice_idle_seconds = int(voice_cfg.get("auto_stop_idle_seconds", 1800))
                self._voice_resume_wait_seconds = int(voice_cfg.get("resume_wait_seconds", 45))
                self._runpod_pod_id = config.get("runpod_pod_id") or voice_cfg.get("runpod_pod_id")
                self._runpod_api_key = config.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY")
                if self.voice_call.available:
                    print("[operator] Live voice call enabled")
                    # Start in silent mode unless explicitly toggled by control file.
                    self.voice_call.playback_enabled = False
                else:
                    self.voice_call = None
            except Exception as e:
                print(f"[operator] Voice call init failed: {e}")

        # Restore last user-selected voice state (default off).
        try:
            if self.voice_control_path.exists():
                with open(self.voice_control_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._voice_active = bool(data.get("voice_active", False))
        except (OSError, json.JSONDecodeError):
            self._voice_active = False

        if self.voice_call:
            self.voice_call.playback_enabled = bool(self._voice_active)

    def _acquire_instance_lock(self) -> None:
        """Prevent multiple operators in the same workspace.

        Moshi's server is single-connection (global lock). Multiple operator instances
        can easily deadlock the voice call handshake and make the system look flaky.
        """
        lock_path = self.workspace / ".capseal" / "operator.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        fp = open(lock_path, "a+")
        try:
            try:
                import fcntl  # Unix only

                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:
                # Best-effort on non-Unix. We still write pid for humans.
                pass
        except OSError as e:
            fp.close()
            if e.errno in (errno.EAGAIN, errno.EACCES):
                raise RuntimeError(
                    f"another capseal operator is already running for workspace {self.workspace}"
                )
            raise

        fp.seek(0)
        fp.truncate()
        fp.write(f"{os.getpid()}\n")
        fp.flush()
        self._instance_lock_fp = fp

    def _init_channels(self, config: dict):
        """Initialize notification channels from config."""
        channels_cfg = config.get("channels", {})

        # Telegram
        tg = channels_cfg.get("telegram", {})
        if tg.get("bot_token") and tg.get("chat_id"):
            from .channels.telegram import TelegramChannel
            self.channels.append(TelegramChannel(
                bot_token=tg["bot_token"],
                chat_id=tg["chat_id"],
                voice_notes=tg.get("voice_notes", False),
                decision_buttons=tg.get("decision_buttons", True),
            ))
            print(f"[operator] Telegram channel configured (chat_id: {tg['chat_id']})")

        # WhatsApp
        wa = channels_cfg.get("whatsapp", {})
        if wa.get("phone_number_id") and wa.get("access_token") and wa.get("recipient"):
            from .channels.whatsapp import WhatsAppChannel
            self.channels.append(WhatsAppChannel(
                phone_number_id=wa["phone_number_id"],
                access_token=wa["access_token"],
                recipient=wa["recipient"],
            ))
            print("[operator] WhatsApp channel configured")

        # iMessage (macOS only)
        im = channels_cfg.get("imessage", {})
        if im.get("recipient"):
            from .channels.imessage import iMessageChannel
            ch = iMessageChannel(recipient=im["recipient"])
            if ch.available:
                self.channels.append(ch)
                print("[operator] iMessage channel configured")
            else:
                print("[operator] iMessage skipped (not macOS)")

        if not self.channels:
            print("[operator] WARNING: No notification channels configured!")
            print("[operator] Set up Telegram: capseal operator --setup telegram")

    async def _sync_voice_toggle(self):
        """Apply TUI voice toggle from .capseal/voice_control.json."""
        target = self._voice_active
        try:
            if self.voice_control_path.exists():
                with open(self.voice_control_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    target = bool(data.get("voice_active", False))
        except (OSError, json.JSONDecodeError):
            return

        if target == self._voice_active:
            return

        if target:
            self._voice_active = True
            if self.voice_call:
                with contextlib.suppress(Exception):
                    await self.voice_call.set_active(True)
            await self._voice_speak("CapSeal Ops online. Monitoring your session.", announce=True)
            print("[operator] Voice toggle: active")
            return

        # Turning OFF: announce once, then mute.
        if self._voice_active:
            await self._voice_speak("CapSeal Ops going silent.", announce=True)
        self._voice_active = False
        if self.voice_call:
            with contextlib.suppress(Exception):
                await self.voice_call.set_active(False)
        print("[operator] Voice toggle: silent")

    async def _voice_speak(self, text: str, *, announce: bool) -> None:
        """Speak via PersonaPlex/Moshi.

        Moshi is speech-to-speech. We inject a short audio prompt (from local tools)
        and let Moshi produce the voiced output. For announcements, wrap with a
        directive to reduce conversational filler.
        """
        if not self._voice_active:
            return
        if not self.voice_call or not self.voice_call.connected:
            return
        cleaned = (text or "").strip()
        if not cleaned:
            return
        payload = cleaned
        if announce:
            payload = f"Announce to the user exactly: {cleaned}"
        async with self._voice_speak_lock:
            with contextlib.suppress(Exception):
                await self.voice_call.speak(payload)

    async def run(self):
        """Main event loop."""
        self.running = True
        print(f"[operator] CapSeal Operator Agent starting")
        print(f"[operator] Workspace: {self.workspace}")
        print(f"[operator] Watching: {self.events_path}")
        print(f"[operator] Notify threshold: {self.config['notify_threshold']}")
        print(f"[operator] Channels: {len(self.channels)} active")
        print()

        # Connect live voice call if configured
        if self.voice_call:
            if self._voice_listen_commands:
                self.voice_call.on_transcript(self._handle_voice_transcript)
                print("[operator] Voice command listening enabled")
            else:
                print("[operator] Voice command listening disabled (narration-only)")
            if await self.voice_call.connect():
                self._voice_listen_task = asyncio.create_task(self.voice_call.listen_loop())
                with contextlib.suppress(Exception):
                    await self.voice_call.set_active(self._voice_active)
            self._voice_maintenance_task = asyncio.create_task(self._voice_maintenance_loop())

        # Send startup notification
        await self._broadcast(Message(
            short_text="\U0001f7e2 CapSeal Operator online. Watching for events.",
            full_text="\U0001f7e2 CapSeal Operator Agent is online and monitoring your workspace.",
            voice_text="CapSeal Ops online. Watching your session. I'll stay quiet unless something interesting happens.",
        ), score=0.5)

        # If events.jsonl already exists, seek to end (unless replay mode)
        if self.events_path.exists() and not self.replay:
            self.last_file_size = self.events_path.stat().st_size
            self.last_file_pos = self.last_file_size
            print(f"[operator] Events file exists ({self.last_file_size} bytes), seeking to end")

        # PTY events: also seek to end on startup to avoid replaying old events
        if self.pty_events_path.exists() and not self.replay:
            self.last_pty_file_size = self.pty_events_path.stat().st_size
            self.last_pty_file_pos = self.last_pty_file_size
            print(f"[operator] PTY events file exists ({self.last_pty_file_size} bytes), seeking to end")
        if self.home_pty_events_path.exists() and self.home_pty_events_path != self.pty_events_path and not self.replay:
            self.last_home_pty_file_size = self.home_pty_events_path.stat().st_size
            self.last_home_pty_file_pos = self.last_home_pty_file_size
            print(f"[operator] Home PTY events file exists ({self.last_home_pty_file_size} bytes), seeking to end")

        elif self.replay:
            print(f"[operator] Replay mode: processing all existing events")

        while self.running:
            try:
                await self._sync_voice_toggle()
                new_events = self._read_new_events()

                for event in new_events:
                    # Dedup by timestamp to prevent duplicate notifications
                    ts = event.get("timestamp") or event.get("ts")
                    if ts and ts in self._seen_timestamps:
                        continue
                    if ts:
                        self._seen_timestamps.add(ts)
                        if len(self._seen_timestamps) > 1000:
                            self._seen_timestamps = set(sorted(self._seen_timestamps)[-500:])
                    await self._process_event(event)

                # Check for incoming commands from channels
                await self._check_incoming()

                # Write operator status for TUI
                self._write_status()

            except Exception as e:
                print(f"[operator] Error in event loop: {e}")

            await asyncio.sleep(0.25)  # 250ms poll

        if self._voice_maintenance_task:
            self._voice_maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._voice_maintenance_task
        if self._voice_listen_task:
            self._voice_listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._voice_listen_task

        # Write offline status on shutdown
        self._write_status(offline=True)

    def _normalize_event(self, event: dict) -> dict:
        """Normalize MCP event format to operator format."""
        # Timestamp: MCP uses "timestamp" (unix float), simulator uses "ts"
        if "timestamp" in event and "ts" not in event:
            event["ts"] = event["timestamp"]

        etype = event.get("type", "")
        data = event.get("data", {}) or {}

        # Event type: MCP emits "seal", daemon expects "session_seal"
        if etype == "seal":
            event["type"] = "session_seal"

        # Gate decision: MCP uses "approve"/"deny", daemon expects "approved"/"denied"
        if etype == "gate":
            d = data.get("decision", "")
            if d == "approve":
                data["decision"] = "approved"
            elif d == "deny":
                data["decision"] = "denied"
            elif d == "flag":
                data["decision"] = "flagged"
            # MCP uses "reason", daemon expects "risk_factors"
            if "reason" in data and "risk_factors" not in data:
                data["risk_factors"] = data["reason"]

        return event

    @staticmethod
    def _normalize_gate_decision(raw: str) -> str:
        value = str(raw or "").strip().lower()
        if value in {"approve", "approved", "pass"}:
            return "approve"
        if value in {"deny", "denied", "skip"}:
            return "deny"
        if value in {"flag", "flagged", "human_review"}:
            return "flag"
        return value

    def _should_force_gate_voice(self, event: dict | None, score: float) -> bool:
        """True when voice output should be forced for a gate event."""
        if not event or event.get("type") != "gate":
            return False

        voice_cfg = self.config.get("voice", {})
        # "enabled" toggles voice synthesis (Phase 5). Live call narration should still work
        # when voice synthesis is disabled, as long as live_call is configured.
        if not (voice_cfg.get("enabled") or voice_cfg.get("live_call")):
            return False
        if not voice_cfg.get("speak_gate_events", True):
            return False

        try:
            min_score = float(voice_cfg.get("speak_min_score", 0.55))
        except (TypeError, ValueError):
            min_score = 0.55
        if score < min_score:
            return False

        decisions = voice_cfg.get("speak_gate_decisions", ["deny", "flag"])
        if not isinstance(decisions, list):
            decisions = ["deny", "flag"]
        allow = {self._normalize_gate_decision(x) for x in decisions}

        decision = self._normalize_gate_decision(event.get("data", {}).get("decision", ""))
        return decision in allow

    async def _process_event(self, event: dict):
        """Process a single event: normalize, update context, score, notify."""
        event = self._normalize_event(event)
        self.context.update(event)
        self._events_processed += 1
        self.last_event_time = time.time()

        score = self.significance.score(event, self.context)
        etype = event.get("type", "unknown")
        data = event.get("data", {})

        # Always log to console
        decision = data.get("decision", "")
        files = data.get("files", [])
        p_fail = data.get("p_fail")

        print(f"[operator] Event: {etype} | score={score:.2f} | "
              f"decision={decision} | files={files} | p_fail={p_fail}")

        if etype == "session_start":
            if self.voice_call and (self._voice_pod_stopped or not self.voice_call.connected):
                if not self._voice_resume_task or self._voice_resume_task.done():
                    self._voice_resume_task = asyncio.create_task(self._resume_voice_on_session_start())

        force_gate_voice = self._should_force_gate_voice(event, score)
        if score >= self.config["notify_threshold"] or force_gate_voice:
            message = self.composer.compose(event, self.context, score)
            await self._broadcast(message, score, event=event)

    def _read_new_events(self) -> list[dict]:
        """Read new lines from events.jsonl and pty_events.jsonl."""
        events = []
        events.extend(self._tail_file(
            self.events_path, "last_file_pos", "last_file_size", reset_context=True))
        events.extend(self._tail_file(
            self.pty_events_path, "last_pty_file_pos", "last_pty_file_size"))
        if self.home_pty_events_path != self.pty_events_path:
            events.extend(self._tail_file(
                self.home_pty_events_path, "last_home_pty_file_pos", "last_home_pty_file_size"))
        return events

    def _tail_file(self, path: Path, pos_attr: str, size_attr: str,
                   reset_context: bool = False) -> list[dict]:
        """Tail a JSONL file using stored position."""
        if not path.exists():
            return []

        try:
            current_size = path.stat().st_size
        except OSError:
            return []

        last_size = getattr(self, size_attr)
        last_pos = getattr(self, pos_attr)

        # File shrunk or replaced — reset
        if current_size < last_size:
            if reset_context:
                print("[operator] Events file reset detected, starting from beginning")
                self.context = SessionContext()
            last_pos = 0

        setattr(self, size_attr, current_size)

        if current_size <= last_pos:
            return []

        events = []
        try:
            with open(path, "r") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                setattr(self, pos_attr, f.tell())
        except OSError as e:
            print(f"[operator] Error reading {path.name}: {e}")

        return events

    async def _broadcast(self, message: Message, score: float, event: dict | None = None):
        """Send notification through all configured channels."""
        tier = self.significance.tier(score)
        force_gate_voice = self._should_force_gate_voice(event, score)
        wants_voice = tier in ("voice_note", "critical") or force_gate_voice
        delivered = False

        # Synthesize voice for voice_note and critical tiers (Phase 5)
        audio = None
        if wants_voice and self.voice and message.voice_text:
            audio = await self.voice.synthesize(message.voice_text)

        # Live voice call narration via PersonaPlex/Moshi.
        if wants_voice and self._voice_active and message.voice_text and self.voice_call and self.voice_call.connected:
            await self._voice_speak(message.voice_text, announce=True)
            delivered = True

        for channel in self.channels:
            try:
                if wants_voice and audio and hasattr(channel, 'send_voice'):
                    await channel.send_voice(audio, caption=message.short_text)
                    delivered = True
                elif wants_voice:
                    await channel.send_text(message.full_text, buttons=message.buttons)
                    delivered = True
                elif score >= self.config["notify_threshold"]:
                    await channel.send_text(message.short_text, buttons=message.buttons)
                    delivered = True
            except Exception as e:
                print(f"[operator] Channel {channel.__class__.__name__} error: {e}")

        if delivered:
            self._last_alert_ts = time.time()

    async def _check_incoming(self):
        """Poll channels for user commands."""
        for channel in self.channels:
            try:
                messages = await channel.poll_messages()
                for msg in messages:
                    await self._handle_command(msg, channel)
            except Exception:
                pass

    async def _handle_command(self, text: str, channel, _depth: int = 0):
        """Handle incoming user commands."""
        if _depth > 1:
            return  # Prevent infinite recursion from NLP

        text_lower = text.strip().lower()

        if text_lower in ("status", "s", "/status"):
            summary = self.context.summary()
            await channel.send_text(f"\U0001f4ca {summary}")

        elif text_lower in ("trust", "/trust"):
            trust = self.context.trust_score
            bar = "\u2588" * int(trust * 10) + "\u2591" * (10 - int(trust * 10))
            await channel.send_text(f"\U0001f6e1\ufe0f Trust: {trust:.0%} [{bar}]")

        elif text_lower in ("files", "/files"):
            if self.context.files_touched:
                lines = ["\U0001f4c1 Files touched:"]
                for f, count in sorted(self.context.files_touched.items(),
                                       key=lambda x: -x[1]):
                    streak = self.context.consecutive_denials.get(f, 0)
                    warn = " \u26a0\ufe0f" if streak > 0 else ""
                    lines.append(f"  {f} ({count}x){warn}")
                await channel.send_text("\n".join(lines))
            else:
                await channel.send_text("\U0001f4c1 No files touched yet.")

        elif text_lower in ("help", "/help"):
            await channel.send_text(
                "\U0001f916 CapSeal Operator Commands:\n"
                "  /status \u2014 Session summary\n"
                "  /trust \u2014 Trust score\n"
                "  /files \u2014 Files touched\n"
                "  /approve [file] \u2014 Override: approve\n"
                "  /deny [file] \u2014 Override: deny\n"
                "  /pause \u2014 Pause session\n"
                "  /resume \u2014 Resume session\n"
                "  /instruct <text> \u2014 Send instruction to agent\n"
                "  /end \u2014 End session\n"
                "  /help \u2014 This message\n\n"
                "Or just type naturally: 'let it through', 'block that', etc."
            )

        elif text_lower.startswith(("/approve", "approve")):
            parts = text.strip().split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else None
            await self.intervention.approve_pending(target=target)
            msg = f"\u2705 Approved{f' for {target}' if target else ''}."
            await channel.send_text(msg)

        elif text_lower.startswith(("/deny", "deny")) and not text_lower.startswith("denied"):
            parts = text.strip().split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else None
            await self.intervention.deny_pending(target=target)
            msg = f"\U0001f6d1 Denied{f' for {target}' if target else ''}."
            await channel.send_text(msg)

        elif text_lower in ("pause", "/pause"):
            await self.intervention.pause_session()
            await channel.send_text("\u23f8 Session paused.")

        elif text_lower in ("resume", "/resume"):
            await self.intervention.resume_session()
            await channel.send_text("\u25b6\ufe0f Session resumed.")

        elif text_lower in ("end", "/end"):
            await self.intervention.end_session()
            await channel.send_text("\u23f9 Session end requested.")

        elif text_lower.startswith(("/instruct", "instruct")):
            parts = text.strip().split(maxsplit=1)
            instruction = parts[1] if len(parts) > 1 else ""
            if instruction:
                await self.intervention.instruct_agent(instruction)
                await channel.send_text(f"\U0001f4e8 Instruction sent: {instruction}")
            else:
                await channel.send_text("\u26a0\ufe0f Usage: /instruct <message to agent>")

        elif text_lower.startswith(("/pty_accept", "pty_accept")):
            await self._inject_pty_input(b"1\n")  # "1" + newline accepts in Claude Code
            await channel.send_text("\u2705 Accepted \u2014 sent Tab to agent.")

        elif text_lower.startswith(("/pty_reject", "pty_reject")):
            await self._inject_pty_input(b"3\n")  # "3" + newline rejects in Claude Code
            await channel.send_text("\u274c Rejected \u2014 sent Escape to agent.")

        elif text_lower.startswith("/investigate"):
            await channel.send_text(
                "\U0001f50d Check the TUI session monitor for full details, "
                "or run: capseal chain <file.cap>"
            )

        elif text_lower.startswith("/diff"):
            parts = text.strip().split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else None
            await channel.send_text(
                f"\U0001f4cb Diff for {target or 'last action'}: "
                f"check the TUI or run: git diff"
            )

        else:
            # NLP command parsing (Phase 8)
            nlp_cfg = self.config.get("nlp", {})
            if nlp_cfg.get("enabled", True):
                try:
                    from .nlp_commands import parse
                    use_llm = nlp_cfg.get("llm_fallback", True)
                    parsed = await parse(text, llm_fallback=use_llm)
                    if parsed:
                        await channel.send_text(f"\U0001f4ac Understood: {parsed}")
                        await self._handle_command(parsed, channel, _depth=_depth + 1)
                        return
                except Exception as e:
                    print(f"[operator] NLP parse error: {e}")

            await channel.send_text("\U0001f914 Unknown command. Try /help")

    async def _inject_pty_input(self, data: bytes):
        """Write bytes to .capseal/pty_input.txt for TUI PTY injection."""
        pty_input_path = self.workspace / ".capseal" / "pty_input.txt"
        pty_input_path.parent.mkdir(parents=True, exist_ok=True)
        pty_input_path.write_bytes(data)
        print(f"[operator] Wrote {len(data)} bytes to pty_input.txt")

    async def _handle_voice_transcript(self, text: str):
        """Handle transcribed speech from live voice call."""
        if not self._voice_listen_commands or not self._voice_active:
            return
        cleaned = text.strip()
        if not cleaned:
            return
        # NOTE: For the Moshi binary protocol, kind=0x02 frames are model output text tokens,
        # not a reliable ASR transcript of the user's microphone. Treat them as "voice text"
        # and only act on explicit slash-commands to avoid accidental interventions.
        print(f"[operator] Voice text: {cleaned}")

        # Only accept explicit slash commands. Freeform voice commands require client-side ASR.
        command = cleaned
        if not command.startswith("/"):
            return

        ack = await self._apply_voice_command(command)
        if not ack:
            return

        if self.channels:
            try:
                await self.channels[0].send_text(f"\U0001f3a4 {ack}")
            except Exception:
                pass

        await self._voice_speak(ack, announce=False)

    async def _apply_voice_command(self, command: str) -> str | None:
        """Apply a slash command parsed from voice transcript."""
        raw = command.strip()
        if not raw:
            return None
        lowered = raw.lower()
        parts = raw.split(maxsplit=1)
        target = parts[1] if len(parts) > 1 else None

        if lowered.startswith("/approve"):
            await self.intervention.approve_pending(target=target)
            return f"Approved{f' for {target}' if target else ''}."
        if lowered.startswith("/deny"):
            await self.intervention.deny_pending(target=target)
            return f"Denied{f' for {target}' if target else ''}."
        if lowered.startswith("/pause"):
            await self.intervention.pause_session()
            return "Session paused."
        if lowered.startswith("/resume"):
            await self.intervention.resume_session()
            return "Session resumed."
        if lowered.startswith("/end"):
            await self.intervention.end_session()
            return "Session end requested."
        if lowered.startswith("/instruct"):
            instruction = target or ""
            if not instruction:
                return "No instruction text provided."
            await self.intervention.instruct_agent(instruction)
            return f"Instruction sent: {instruction}"
        if lowered.startswith("/status"):
            return self.context.summary()
        if lowered.startswith("/trust"):
            return f"Trust score {self.context.trust_score:.0%}."
        if lowered.startswith("/files"):
            if not self.context.files_touched:
                return "No files touched yet."
            top = sorted(self.context.files_touched.items(), key=lambda x: -x[1])[:3]
            details = ", ".join(f"{name} ({count}x)" for name, count in top)
            return f"Top files: {details}."
        if lowered.startswith("/help"):
            return "Say approve, deny, pause, resume, status, trust, files, or instruct."

        return None

    async def _voice_maintenance_loop(self):
        """Reconnect voice call and auto-stop idle RunPod pod."""
        while self.running:
            try:
                await asyncio.sleep(30)
                if not self.voice_call:
                    continue

                if self.voice_call.connected and self._voice_idle_seconds > 0:
                    idle = time.time() - self.last_event_time
                    if idle >= self._voice_idle_seconds:
                        print(f"[operator] Voice idle for {int(idle)}s, stopping pod")
                        await self._auto_stop_voice()
                        continue

                # Best-effort reconnect if pod is expected online.
                if (
                    not self._voice_pod_stopped
                    and not self.voice_call.connected
                    and (time.time() - self._last_voice_reconnect_attempt) >= self._voice_reconnect_interval_seconds
                ):
                    self._last_voice_reconnect_attempt = time.time()
                    ok = await self.voice_call.connect()
                    if ok:
                        if not self._voice_listen_task or self._voice_listen_task.done():
                            self._voice_listen_task = asyncio.create_task(self.voice_call.listen_loop())
                        with contextlib.suppress(Exception):
                            await self.voice_call.set_active(self._voice_active)
                        if self._voice_active:
                            await self._voice_speak("CapSeal Ops reconnected.", announce=True)
            except Exception as e:
                print(f"[operator] voice maintenance error: {e}")

    async def _auto_stop_voice(self):
        if not self.voice_call:
            return
        if self.voice_call.connected:
            await self._voice_speak("Going offline due to inactivity.", announce=True)
            await asyncio.sleep(0.2)
        await self.voice_call.disconnect()
        self._voice_pod_stopped = True

        if self._runpod_pod_id and self._runpod_api_key:
            try:
                from .runpod_ops import stop_pod
                stop_pod(self._runpod_api_key, self._runpod_pod_id)
                print(f"[operator] stopped RunPod pod {self._runpod_pod_id}")
            except Exception as e:
                print(f"[operator] failed to stop pod: {e}")

    async def _resume_voice_on_session_start(self):
        if not self.voice_call:
            return
        if self.voice_call.connected:
            return

        if self._runpod_pod_id and self._runpod_api_key and self._voice_pod_stopped:
            try:
                from .runpod_ops import resume_pod
                resume_pod(self._runpod_api_key, self._runpod_pod_id)
                print(f"[operator] resumed RunPod pod {self._runpod_pod_id}")
                await asyncio.sleep(max(5, self._voice_resume_wait_seconds))
                self._voice_pod_stopped = False
            except Exception as e:
                print(f"[operator] failed to resume pod: {e}")
                return

        ok = await self.voice_call.connect()
        if ok:
            with contextlib.suppress(Exception):
                await self.voice_call.set_active(self._voice_active)
            if self._voice_listen_commands:
                self.voice_call.on_transcript(self._handle_voice_transcript)
            if not self._voice_listen_task or self._voice_listen_task.done():
                self._voice_listen_task = asyncio.create_task(self.voice_call.listen_loop())
            if self._voice_active:
                await self._voice_speak("CapSeal Ops back online. Monitoring your session.", announce=True)

    @staticmethod
    def _channel_type_name(channel) -> str:
        """Normalize channel class names into stable display values."""
        name = channel.__class__.__name__.strip()
        lowered = name.lower()
        if lowered.endswith("channel"):
            lowered = lowered[: -len("channel")]
        # Normalize common aliases for UI readability.
        aliases = {
            "telegram": "telegram",
            "whatsapp": "whatsapp",
            "imessage": "imessage",
        }
        return aliases.get(lowered, lowered)

    def _write_status(self, offline: bool = False):
        """Write operator status file atomically for TUI consumption."""
        now = time.time()
        if not offline and now - self._last_status_write < 5.0:
            return
        self._last_status_write = now

        status = {
            "online": not offline,
            "workspace": str(self.workspace),
            "channels": len(self.channels),
            "channel_types": [self._channel_type_name(ch) for ch in self.channels],
            "session_id": self.context.session_id,
            "total_actions": self.context.total_actions,
            "events_processed": self._events_processed,
            "voice_connected": bool(self.voice_call and self.voice_call.connected),
            "voice_active": bool(self._voice_active),
            "last_alert_ts": self._last_alert_ts,
            "notify_threshold": self.config.get("notify_threshold", 0.5),
            "ts": now,
        }

        try:
            for path in (self.status_path, self.home_status_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(".tmp")
                with open(tmp_path, "w") as f:
                    json.dump(status, f)
                tmp_path.rename(path)
        except OSError:
            pass


def parse_args():
    """Simple arg parser (no external deps needed)."""
    import argparse
    parser = argparse.ArgumentParser(description="CapSeal Operator Agent")
    parser.add_argument("--workspace", "-w", type=Path, default=Path("."),
                        help="Workspace directory to monitor")
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="Config file path (default: ~/.capseal/operator.json)")
    parser.add_argument("--setup", type=str, choices=["telegram", "whatsapp"],
                        help="Interactive channel setup")
    parser.add_argument("--test", action="store_true",
                        help="Send a test notification and exit")
    parser.add_argument("--replay", action="store_true",
                        help="Replay all existing events instead of seeking to end")
    return parser.parse_args()


async def interactive_setup_telegram():
    """Walk user through Telegram bot setup."""
    print("=" * 60)
    print("  CapSeal Operator \u2014 Telegram Setup")
    print("=" * 60)
    print()
    print("Step 1: Open Telegram and message @BotFather")
    print("Step 2: Send /newbot and follow the prompts")
    print("Step 3: Copy the bot token you receive")
    print()

    print("Security note: storing tokens in operator.json is convenient but plaintext.")
    print("Recommended: store token in env var CAPSEAL_TELEGRAM_BOT_TOKEN.")
    store_plaintext = input("Store token in ~/.capseal/operator.json? [y/N]: ").strip().lower() in {"y", "yes"}
    bot_token = input("Paste your bot token: ").strip()
    if not bot_token:
        print("No token provided, aborting.")
        return

    print()
    print("Step 4: Open a chat with your new bot in Telegram")
    print("Step 5: Send any message to the bot (e.g., 'hello')")
    print()
    input("Press Enter after you've sent a message to the bot...")

    # Fetch chat_id
    import urllib.request
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
            if data.get("ok") and data.get("result"):
                chat_id = str(data["result"][-1]["message"]["chat"]["id"])
                print(f"\n\u2705 Found your chat_id: {chat_id}")
            else:
                print("\n\u274c No messages found. Make sure you sent a message to the bot.")
                return
    except Exception as e:
        print(f"\n\u274c Error fetching chat_id: {e}")
        return

    # Save config
    config_dir = Path.home() / ".capseal"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "operator.json"

    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    config.setdefault("channels", {})
    config["channels"]["telegram"] = {
        "bot_token": bot_token if store_plaintext else None,
        "bot_token_env": None if store_plaintext else "CAPSEAL_TELEGRAM_BOT_TOKEN",
        "chat_id": chat_id,
        "voice_notes": False,
        "decision_buttons": True,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n\u2705 Config saved to {config_path}")
    if not store_plaintext:
        print("\nSet this before running the operator:")
        print("  export CAPSEAL_TELEGRAM_BOT_TOKEN='<your-bot-token>'")
    print(f"\nRun: capseal operator /your/project --test")


async def send_test(workspace: Path, config: dict):
    """Send a test notification to verify setup."""
    daemon = OperatorDaemon(workspace, config)
    if not daemon.channels:
        print("\u274c No channels configured. Run: capseal operator --setup telegram")
        return

    print("Sending test notification...")
    await daemon._broadcast(Message(
        short_text="\U0001f9ea CapSeal Operator test \u2014 if you see this, you're set!",
        full_text="\U0001f9ea CapSeal Operator test notification.\n\nIf you're reading this, your operator is configured correctly. Start a CapSeal session and the operator will notify you of important events.",
        voice_text="This is a test from CapSeal Ops. If you hear this, everything is working.",
    ), score=0.5)
    print("\u2705 Test notification sent!")


async def main():
    args = parse_args()

    if args.setup == "telegram":
        await interactive_setup_telegram()
        return

    config = load_config(args.config, workspace=args.workspace)

    if args.test:
        await send_test(args.workspace, config)
        return

    daemon = OperatorDaemon(args.workspace.resolve(), config)

    if args.replay:
        daemon.replay = True

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: setattr(daemon, 'running', False))

    try:
        await daemon.run()
    except KeyboardInterrupt:
        pass

    print("\n[operator] Shutting down.")
    for channel in daemon.channels:
        try:
            await channel.send_text("\U0001f534 CapSeal Operator going offline.")
        except Exception:
            pass

    if daemon.voice_call:
        await daemon.voice_call.disconnect()


def main_sync():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
