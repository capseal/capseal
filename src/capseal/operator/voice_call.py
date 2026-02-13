"""CapSeal Operator live voice channel.

Supports:
1) Moshi/PersonaPlex binary protocol over `/api/chat` (preferred)
2) Legacy JSON stream protocol (`.../v1/stream`) for backward compatibility
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import json
import re
import shutil
import subprocess
import threading
import time
from typing import Awaitable, Callable, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit, urlunsplit, urlunparse

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


def _run_capture_text(args: list[str], *, timeout_s: float = 2.0) -> str | None:
    try:
        cp = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    return (cp.stdout or "").strip()


def _parse_boolish(value: str | None) -> bool | None:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    # Handle "key = true" and similar.
    if "=" in lowered:
        tail = lowered.split("=", 1)[1].strip()
        if tail in {"true", "1", "yes", "on"}:
            return True
        if tail in {"false", "0", "no", "off"}:
            return False
    return None


VOICE_PROFILES = {
    # Fastest response, higher chance of choppiness under jitter.
    "low_latency": {
        "uplink_chunk_bytes": 1024,
        "uplink_sleep_seconds": 0.004,
        "playback_flush_every_frames": 2,
        "playback_backend": "mpv",
        "mpv_cache_secs": 0.08,
        "mpv_readahead_secs": 0.08,
        "ffplay_low_delay": True,
    },
    # Best practical compromise for live narration.
    "balanced": {
        "uplink_chunk_bytes": 1536,
        "uplink_sleep_seconds": 0.006,
        "playback_flush_every_frames": 4,
        "playback_backend": "mpv",
        "mpv_cache_secs": 0.18,
        "mpv_readahead_secs": 0.18,
        "ffplay_low_delay": False,
    },
    # Smoothest and most stable, adds perceptible delay.
    "high_quality": {
        "uplink_chunk_bytes": 3072,
        "uplink_sleep_seconds": 0.012,
        "playback_flush_every_frames": 8,
        "playback_backend": "mpv",
        "mpv_cache_secs": 0.35,
        "mpv_readahead_secs": 0.35,
        "ffplay_low_delay": False,
    },
    # Alias for "as low latency as possible without trashing quality".
    "hq_low_latency": {
        "uplink_chunk_bytes": 1280,
        "uplink_sleep_seconds": 0.005,
        "playback_flush_every_frames": 3,
        "playback_backend": "mpv",
        "mpv_cache_secs": 0.12,
        "mpv_readahead_secs": 0.12,
        "ffplay_low_delay": False,
    },
}


def _normalize_moshi_ws_url(url: str) -> str:
    """Convert base pod URL to Moshi chat endpoint."""
    parsed = urlparse(url)
    if not parsed.scheme:
        # Assume secure endpoint when scheme omitted.
        parsed = urlparse("wss://" + url.lstrip("/"))
    if parsed.path.endswith("/api/chat"):
        path = parsed.path
    else:
        path = "/api/chat"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def _augment_moshi_query(
    url: str,
    *,
    voice_preset: str,
    silence_guard_enabled: bool,
    silence_guard_text: str,
) -> str:
    """Ensure required Moshi query params exist and optionally add a silence guard prompt."""
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)

    # Ensure a voice prompt is selected (PersonaPlex expects *.pt file names).
    query.setdefault("voice_prompt", [f"{voice_preset}.pt"])

    if silence_guard_enabled:
        base = (query.get("text_prompt") or [""])[0] or ""
        extra = (silence_guard_text or "").strip()
        if extra and extra.lower() not in base.lower():
            base = (base.strip() + " " + extra).strip()
        query["text_prompt"] = [base]

    new_query = urlencode(query, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


class VoiceCallManager:
    """Manages a live WebSocket voice channel."""

    def __init__(self, config: dict):
        self.config = config
        self.ws_url = str(config.get("personaplex_ws_url", "wss://api.personaplex.io/v1/stream"))
        self.voice_preset = config.get("voice_preset", "NATM1")
        self.protocol = self._resolve_protocol(str(config.get("protocol", "")).strip().lower())
        self.voice_profile = str(config.get("voice_profile", "balanced")).strip().lower()
        if self.voice_profile not in VOICE_PROFILES:
            self.voice_profile = "balanced"
        profile = dict(VOICE_PROFILES[self.voice_profile])
        # Lazily initialized inside async methods to avoid loop-bound issues.
        self._connect_lock: asyncio.Lock | None = None
        self.ws = None
        self.connected = False
        self.audio_frames_received = 0
        self.audio_bytes_received = 0
        self._debug_downlink_frames = int(config.get("debug_downlink_frames", 0) or 0)
        self._debug_downlink_logged = 0
        self._debug_uplink_frames = int(config.get("debug_uplink_frames", 0) or 0)
        self._debug_uplink_logged = 0
        self._on_transcript: Optional[Callable[[str], Awaitable[None]]] = None
        self._text_buffer: str = ""
        self.playback_enabled = bool(config.get("playback_enabled", True))
        self.playback_backend = str(config.get("playback_backend", profile["playback_backend"])).strip().lower()
        self.uplink_chunk_bytes = int(config.get("uplink_chunk_bytes", profile["uplink_chunk_bytes"]))
        self.uplink_sleep_seconds = float(config.get("uplink_sleep_seconds", profile["uplink_sleep_seconds"]))
        self.mpv_cache_secs = float(config.get("mpv_cache_secs", profile["mpv_cache_secs"]))
        self.mpv_readahead_secs = float(config.get("mpv_readahead_secs", profile["mpv_readahead_secs"]))
        self.ffplay_low_delay = bool(config.get("ffplay_low_delay", profile["ffplay_low_delay"]))
        self._player_proc: subprocess.Popen | None = None
        self._player_lock = threading.Lock()
        self._player_bytes = 0
        self._player_name = ""
        self._player_stderr_thread: threading.Thread | None = None
        self._playback_restart_count = 0
        self._playback_frame_counter = 0
        self._playback_flush_every_frames = int(
            config.get("playback_flush_every_frames", profile["playback_flush_every_frames"])
        )
        self._send_lock = asyncio.Lock()
        self.freeform_enabled = bool(config.get("freeform", False))
        self.mic_enabled = bool(config.get("mic_enabled", self.freeform_enabled))
        self.mic_input = str(config.get("mic_input", "default")).strip() or "default"
        self.mic_chunk_bytes = int(config.get("mic_chunk_bytes", self.uplink_chunk_bytes))
        self.mic_backend = str(config.get("mic_backend", "auto")).strip().lower()
        self.mic_sample_rate = int(config.get("mic_sample_rate", 24000))
        self.mic_frame_duration_ms = int(config.get("mic_frame_duration_ms", 20))
        # Bluetooth UX: if output is Bluetooth (A2DP), opening a Bluetooth *mic* often forces HFP/HSP
        # which can mute output and/or degrade audio. Default behavior is to avoid BT mic when possible.
        self.bt_avoid_headset_mic = bool(config.get("bt_avoid_headset_mic", True))
        # Best-effort: temporarily disable WirePlumber autoswitch-to-headset-profile while voice is active.
        # We restore the previous value when stopping mic uplink.
        self.bt_disable_autoswitch = bool(config.get("bt_disable_autoswitch", True))
        self._wpctl_prev_autoswitch: bool | None = None
        self._wpctl_autoswitch_changed = False
        self._audio_setup_logged = False
        # Local VAD/noise gate: replace low-energy mic frames with zeros (silence) before Opus encode.
        # This preserves timing (silence still flows) while stopping the model from responding to ambient noise.
        self.mic_vad_enabled = bool(config.get("mic_vad_enabled", True))
        self.mic_vad_db = float(config.get("mic_vad_db", -45.0))
        self.mic_vad_hangover_ms = int(config.get("mic_vad_hangover_ms", 240))
        self._vad_open_frames = 0
        self._mic_capture_proc: subprocess.Popen | None = None
        self._mic_encode_proc: subprocess.Popen | None = None
        self._mic_pcm_task: asyncio.Task | None = None
        self._mic_ogg_task: asyncio.Task | None = None
        self._mic_uplink_running = False
        # Moshi OpusStreamReader expects a single OGG/Opus byte stream.
        # We must not interleave mic OGG bytes with separate speak() OGG payloads.
        self._speak_uplink_active = False
        # Echo/feedback suppression: if the local mic is picking up downlink playback,
        # the model can end up "talking to itself". Duck mic uplink while downlink audio is active.
        self.mic_duck_downlink = bool(config.get("mic_duck_downlink", True))
        self.mic_duck_hold_ms = int(config.get("mic_duck_hold_ms", 350))
        self._downlink_last_audio_ts = 0.0
        self.silence_guard_enabled = bool(config.get("silence_guard_enabled", True))
        self.silence_guard_text = str(
            config.get(
                "silence_guard_text",
                (
                    "You are CapSeal Ops, a security operations monitor.\n"
                    "RULES:\n"
                    "- Do not initiate conversation.\n"
                    "- During silence or background noise, stay silent. Do not fill pauses.\n"
                    "- Never generate greetings, pleasantries, or filler.\n"
                    "- When you hear an announcement instruction (\"Announce to the user exactly\"),\n"
                    "  repeat the words after the colon verbatim and then stop.\n"
                    "- Stay focused on the live session.\n"
                ),
            )
        ).strip()

    @staticmethod
    def _guess_protocol(ws_url: str) -> str:
        lowered = ws_url.lower()
        if "/v1/stream" in lowered or lowered.endswith("/stream"):
            return "json_stream"
        return "moshi_binary"

    def _resolve_protocol(self, raw: str) -> str:
        if raw in {"", "auto"}:
            return self._guess_protocol(self.ws_url)
        if raw in {"moshi", "moshi_binary", "binary"}:
            return "moshi_binary"
        if raw in {"json", "json_stream", "stream"}:
            return "json_stream"
        return self._guess_protocol(self.ws_url)

    @property
    def available(self) -> bool:
        return HAS_WEBSOCKETS

    async def connect(self) -> bool:
        """Open WebSocket connection."""
        if not HAS_WEBSOCKETS:
            print("[voice_call] websockets not installed. pip install websockets")
            return False

        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            return await self._connect_inner()

    async def _connect_inner(self) -> bool:
        """Inner connect implementation (serialized by _connect_lock)."""
        # Ensure we never leak multiple open sockets. Moshi holds a global lock per
        # connection, so leaked/duplicate websockets can block new handshakes.
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.stop_mic_uplink()
            with contextlib.suppress(Exception):
                await self.ws.close()
            self.ws = None
            self.connected = False

        if self.protocol == "json_stream":
            target = self.ws_url
        else:
            target = _augment_moshi_query(
                _normalize_moshi_ws_url(self.ws_url),
                voice_preset=self.voice_preset,
                silence_guard_enabled=self.silence_guard_enabled,
                silence_guard_text=self.silence_guard_text,
            )
        try:
            # Moshi's aiohttp WS endpoint can be sensitive to client ping/pong keepalive.
            # Disable protocol-level ping for binary mode and rely on reconnect loop.
            ping_interval = 20
            ping_timeout = 20
            if self.protocol == "moshi_binary":
                ping_interval = None
                ping_timeout = None

            self.ws = await websockets.connect(
                target,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                open_timeout=15,
                close_timeout=5,
                max_size=None,
            )

            if self.protocol == "json_stream":
                await self.ws.send(json.dumps({"type": "config", "voice_preset": self.voice_preset}))
            else:
                # Moshi server sends handshake byte 0x00 on successful connect.
                handshake_timeout = float(self.config.get("handshake_timeout_seconds", 45))
                handshake = await asyncio.wait_for(self.ws.recv(), timeout=handshake_timeout)
                if isinstance(handshake, (bytes, bytearray)) and len(handshake) > 0 and handshake[0] == 0x00:
                    pass
                else:
                    print("[voice_call] warning: unexpected handshake frame")
                if self.playback_enabled:
                    self._start_local_player()
                print(f"[voice_call] profile={self.voice_profile}")

            self.connected = True
            if self.playback_enabled and self.protocol == "moshi_binary" and self.mic_enabled:
                await self.start_mic_uplink()
            print(f"[voice_call] connected ({self.protocol})")
            return True
        except Exception as e:
            print(f"[voice_call] connection failed: {e!r}")
            with contextlib.suppress(Exception):
                await self.stop_mic_uplink()
            if self.ws is not None:
                with contextlib.suppress(Exception):
                    await self.ws.close()
            self.connected = False
            self.ws = None
            return False

    async def speak(self, text: str) -> None:
        """Speak text on the live call.

        For Moshi binary protocol, this converts text to Opus bytes locally
        (`espeak` + `ffmpeg`) and injects them into the *existing* mic uplink
        encoder so the server receives a single continuous OGG/Opus byte stream.
        """
        if not self.connected or not self.ws:
            return
        text = (text or "").strip()
        if not text:
            return

        try:
            if self.protocol == "json_stream":
                async with self._send_lock:
                    await self.ws.send(json.dumps({"type": "speak", "text": text}))
                return

            # Ensure the uplink pipeline is running so the server sees one continuous stream.
            if not self._mic_uplink_running:
                await self.start_mic_uplink()

            pcm = await asyncio.to_thread(self._text_to_pcm_s16le, text[:800])
            if not pcm:
                print("[voice_call] skipped speak: failed to synthesize pcm input")
                return

            # Prevent mic frames from being written to the encoder while we inject the prompt.
            self._speak_uplink_active = True
            await self._inject_pcm_to_encoder(pcm)
            print(f"[voice_call] injected prompt pcm ({len(pcm)} bytes)")
        except Exception as e:
            print(f"[voice_call] speak error: {e}")
            self.connected = False
        finally:
            self._speak_uplink_active = False

    def _text_to_opus_ogg(self, text: str) -> bytes:
        """Offline text -> OGG/Opus using local tools.

        This avoids API keys and works with Moshi's OpusStreamReader input.
        """
        espeak = shutil.which("espeak")
        ffmpeg = shutil.which("ffmpeg")
        if not espeak or not ffmpeg:
            return b""

        # Step 1: text -> wav bytes
        espeak_cp = subprocess.run(
            [espeak, "--stdout", "--stdin"],
            input=text.encode("utf-8", errors="ignore"),
            capture_output=True,
            check=False,
        )
        if espeak_cp.returncode != 0 or not espeak_cp.stdout:
            return b""

        # Step 2: wav -> ogg/opus bytes at 24k mono
        ffmpeg_cp = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "libopus",
                "-application",
                "voip",
                "-frame_duration",
                "20",
                "-f",
                "ogg",
                "pipe:1",
            ],
            input=espeak_cp.stdout,
            capture_output=True,
            check=False,
        )
        if ffmpeg_cp.returncode != 0:
            return b""
        return ffmpeg_cp.stdout or b""

    def _text_to_pcm_s16le(self, text: str) -> bytes:
        """Offline text -> s16le PCM (mono) at mic_sample_rate.

        We inject PCM into the existing encoder stdin to keep a single OGG/Opus stream.
        """
        espeak = shutil.which("espeak")
        ffmpeg = shutil.which("ffmpeg")
        if not espeak or not ffmpeg:
            return b""

        espeak_cp = subprocess.run(
            [espeak, "--stdout", "--stdin"],
            input=text.encode("utf-8", errors="ignore"),
            capture_output=True,
            check=False,
        )
        if espeak_cp.returncode != 0 or not espeak_cp.stdout:
            return b""

        ffmpeg_cp = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(self.mic_sample_rate),
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=espeak_cp.stdout,
            capture_output=True,
            check=False,
        )
        if ffmpeg_cp.returncode != 0:
            return b""
        return ffmpeg_cp.stdout or b""

    async def _inject_pcm_to_encoder(self, pcm: bytes) -> None:
        """Write PCM bytes into the encoder stdin with roughly real-time pacing."""
        enc = self._mic_encode_proc
        if not enc or not enc.stdin:
            return

        bytes_per_sample = 2
        samples_per_frame = int(self.mic_sample_rate * (self.mic_frame_duration_ms / 1000.0))
        frame_bytes = max(1, samples_per_frame * bytes_per_sample)
        flush_every_frames = max(1, int(self.config.get("mic_flush_every_frames", 3)))
        frames_since_flush = 0
        loop = asyncio.get_running_loop()

        # Pad to full frames so timing stays consistent.
        if len(pcm) % frame_bytes:
            pcm = pcm + (b"\x00" * (frame_bytes - (len(pcm) % frame_bytes)))

        for off in range(0, len(pcm), frame_bytes):
            if not (self.connected and self._mic_uplink_running):
                break
            chunk = pcm[off : off + frame_bytes]
            await loop.run_in_executor(None, enc.stdin.write, chunk)
            frames_since_flush += 1
            if frames_since_flush >= flush_every_frames:
                frames_since_flush = 0
                await loop.run_in_executor(None, enc.stdin.flush)
            await asyncio.sleep(self.mic_frame_duration_ms / 1000.0)

        # Small silence tail so the model can detect end-of-utterance.
        tail_frames = max(
            1,
            int(int(self.config.get("mic_silence_tail_ms", 200)) / max(1, self.mic_frame_duration_ms)),
        )
        tail_frames = min(tail_frames, 12)
        for _ in range(tail_frames):
            if not (self.connected and self._mic_uplink_running):
                break
            await loop.run_in_executor(None, enc.stdin.write, b"\x00" * frame_bytes)
            frames_since_flush += 1
            if frames_since_flush >= flush_every_frames:
                frames_since_flush = 0
                await loop.run_in_executor(None, enc.stdin.flush)
            await asyncio.sleep(self.mic_frame_duration_ms / 1000.0)

    async def listen_loop(self) -> None:
        """Background receive loop for transcripts/audio markers."""
        if not self.ws:
            return
        try:
            async for message in self.ws:
                if self.protocol == "json_stream":
                    await self._handle_json_message(message)
                else:
                    await self._handle_binary_message(message)
        except Exception as e:
            print(f"[voice_call] listen loop ended: {e}")
        finally:
            self.connected = False

    async def _handle_json_message(self, message: object) -> None:
        if not isinstance(message, str):
            return
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if data.get("type") == "transcript" and self._on_transcript:
            await self._on_transcript(str(data.get("text", "")))

    async def _handle_binary_message(self, message: object) -> None:
        if not isinstance(message, (bytes, bytearray)) or len(message) == 0:
            return
        kind = message[0]
        payload = bytes(message[1:])

        if kind == 0x00:
            return  # handshake
        if kind == 0x01:
            self.audio_frames_received += 1
            self.audio_bytes_received += len(payload)
            self._downlink_last_audio_ts = time.time()
            if self._debug_downlink_frames and self._debug_downlink_logged < self._debug_downlink_frames:
                self._debug_downlink_logged += 1
                sig = payload[:4]
                sig_ascii = ""
                if sig and all(32 <= b < 127 for b in sig):
                    sig_ascii = sig.decode("ascii", errors="ignore")
                print(
                    "[voice_call] downlink "
                    f"{self._debug_downlink_logged}/{self._debug_downlink_frames} "
                    f"len={len(payload)} sig={sig_ascii or sig!r} first16={payload[:16].hex()}"
                )
            self._write_playback(payload)
            if self.playback_enabled and self.audio_frames_received % 10 == 0:
                print(
                    "[voice_call] recv audio "
                    f"frames={self.audio_frames_received} bytes={self.audio_bytes_received}"
                )
            return
        if kind == 0x02:
            token = self._normalize_transcript_text(payload.decode("utf-8", errors="ignore"))
            if not token:
                return
            self._text_buffer += token
            # Flush on likely sentence boundaries for command parsing.
            if any(ch in token for ch in ".!?\n") or len(self._text_buffer) >= 120:
                text = self._normalize_transcript_text(self._text_buffer.strip())
                self._text_buffer = ""
                if text and self._on_transcript:
                    await self._on_transcript(text)
            return
        if kind == 0x05:
            err = payload.decode("utf-8", errors="ignore")
            print(f"[voice_call] server error: {err}")
            return
        if kind == 0x06:
            return
        print(f"[voice_call] unknown frame kind={kind}")

    def on_transcript(self, callback: Optional[Callable[[str], Awaitable[None]]]) -> None:
        """Register callback for incoming transcript chunks."""
        self._on_transcript = callback

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        """Best-effort cleanup for transcript artifacts from streaming chunks."""
        if not text:
            return ""
        cleaned = text.replace("\u2019", "'").replace("\u2018", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        # Common dropped-letter contractions seen in streaming transcript output.
        cleaned = re.sub(r"\bIt'\s", "It's ", cleaned)
        cleaned = re.sub(r"\bI'\s", "I've ", cleaned)
        cleaned = re.sub(r"\bYou'\s", "You're ", cleaned)
        cleaned = re.sub(r"\bWe'\s", "We're ", cleaned)
        cleaned = re.sub(r"\bThey'\s", "They're ", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned

    async def disconnect(self) -> None:
        """Close websocket."""
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            await self.stop_mic_uplink()
            if self.ws:
                with contextlib.suppress(Exception):
                    await self.ws.close()
            self.ws = None
            self._stop_local_player()
            self.connected = False

    async def set_active(self, active: bool) -> None:
        """Apply voice toggle to playback and freeform uplink."""
        self.playback_enabled = bool(active)
        if not active:
            await self.stop_mic_uplink()
            self._stop_local_player()
            return
        if self.connected and self.playback_enabled and not self._player_proc:
            self._start_local_player()
        if self.connected and self.protocol == "moshi_binary" and self.mic_enabled:
            await self.start_mic_uplink()

    def _build_mic_ffmpeg_cmd(self, backend: str) -> list[str] | None:
        # Backward-compat helper: retained for callers that still expect a single ffmpeg command.
        # Prefer the PCM->VAD->OGG pipeline started by _start_mic_uplink_processes().
        return self._build_mic_capture_cmd(backend)

    def _build_mic_capture_cmd(self, backend: str) -> list[str] | None:
        """Capture raw PCM from mic (s16le) for local VAD/noise gating."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None
        if backend == "pulse":
            input_args = ["-f", "pulse", "-i", self.mic_input]
        elif backend == "alsa":
            input_args = ["-f", "alsa", "-i", self.mic_input]
        else:
            return None
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            *input_args,
            "-ac",
            "1",
            "-ar",
            str(self.mic_sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]

    def _build_mic_encode_cmd(self) -> list[str] | None:
        """Encode raw PCM (s16le) to OGG/Opus bytes for Moshi OpusStreamReader."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(self.mic_sample_rate),
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-application",
            "voip",
            "-frame_duration",
            str(self.mic_frame_duration_ms),
            "-f",
            "ogg",
            "pipe:1",
        ]

    def _start_mic_uplink_processes(self) -> bool:
        """Start mic capture + encoder processes.

        We capture PCM for VAD/noise gating, then re-encode to OGG/Opus to keep a valid stream.
        """
        if (
            self._mic_capture_proc
            and self._mic_capture_proc.poll() is None
            and self._mic_encode_proc
            and self._mic_encode_proc.poll() is None
        ):
            return True

        backends: list[str]
        if self.mic_backend == "auto":
            backends = ["pulse", "alsa"]
        elif self.mic_backend in {"pulse", "alsa"}:
            backends = [self.mic_backend]
        else:
            backends = ["pulse", "alsa"]

        for backend in backends:
            try:
                capture_cmd = self._build_mic_capture_cmd(backend)
                encode_cmd = self._build_mic_encode_cmd()
                if not capture_cmd or not encode_cmd:
                    continue

                self._mic_capture_proc = subprocess.Popen(
                    capture_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self._mic_encode_proc = subprocess.Popen(
                    encode_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.mic_backend = backend
                print(f"[voice_call] mic uplink ready ({backend}:{self.mic_input})")
                return True
            except Exception as e:
                print(f"[voice_call] mic uplink start failed ({backend}): {e}")
                self._mic_capture_proc = None
                self._mic_encode_proc = None
        return False

    async def start_mic_uplink(self) -> None:
        """Stream local microphone audio to Moshi for full freeform conversation."""
        if not self.connected or not self.ws or self.protocol != "moshi_binary" or not self.mic_enabled:
            return
        if self._mic_pcm_task and not self._mic_pcm_task.done():
            return
        self._maybe_avoid_bluetooth_mic()
        self._maybe_disable_bt_autoswitch()
        if not self._start_mic_uplink_processes():
            print("[voice_call] mic uplink unavailable (ffmpeg/input device)")
            return
        self._mic_uplink_running = True
        self._mic_pcm_task = asyncio.create_task(self._mic_pcm_to_encoder_loop())
        self._mic_ogg_task = asyncio.create_task(self._mic_encoder_to_ws_loop())
        print("[voice_call] mic uplink started")

    async def stop_mic_uplink(self) -> None:
        self._mic_uplink_running = False
        for task_attr in ("_mic_pcm_task", "_mic_ogg_task"):
            task = getattr(self, task_attr)
            if not task:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            setattr(self, task_attr, None)

        for proc_attr in ("_mic_capture_proc", "_mic_encode_proc"):
            proc = getattr(self, proc_attr)
            setattr(self, proc_attr, None)
            if not proc:
                continue
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print("[voice_call] mic uplink stopped")
        self._maybe_restore_bt_autoswitch()

    def _maybe_avoid_bluetooth_mic(self) -> None:
        """If output is Bluetooth, prefer a non-Bluetooth mic source to avoid A2DP->HFP switch."""
        if not self.bt_avoid_headset_mic:
            return
        if self.mic_backend not in {"auto", "pulse"}:
            return
        pactl = shutil.which("pactl")
        if not pactl:
            return

        sink = _run_capture_text([pactl, "get-default-sink"])
        source = _run_capture_text([pactl, "get-default-source"])
        if not sink:
            return
        output_is_bt = "bluez" in sink.lower()
        if not output_is_bt:
            return

        # Log audio situation once per instance.
        if not self._audio_setup_logged:
            self._audio_setup_logged = True
            print(f"[voice_call] audio output is bluetooth (sink={sink})")

        # Respect explicit mic_input, unless it looks like a BT source.
        configured = (self.mic_input or "").strip()
        configured_is_default = configured.lower() == "default"
        configured_is_bt = "bluez" in configured.lower()
        default_source_is_bt = bool(source and "bluez" in source.lower())
        if not (configured_is_default or configured_is_bt or default_source_is_bt):
            return

        sources_raw = _run_capture_text([pactl, "list", "sources", "short"], timeout_s=3.0)
        if not sources_raw:
            return
        sources: list[str] = []
        for line in sources_raw.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                sources.append(parts[1].strip())

        # Drop monitors and BT sources.
        candidates = [
            s
            for s in sources
            if s
            and "monitor" not in s.lower()
            and "bluez" not in s.lower()
        ]
        if not candidates:
            print("[voice_call] warning: bluetooth output detected but no non-bt mic sources found; using default mic")
            return

        def score(name: str) -> tuple[int, int]:
            # Higher is better. Prefer USB, then analog, then anything else.
            lowered = name.lower()
            usb = 2 if "usb" in lowered else 0
            analog = 1 if ("analog" in lowered or "pci" in lowered) else 0
            return (usb, analog)

        best = sorted(candidates, key=score, reverse=True)[0]
        if best and best != configured:
            self.mic_input = best
            print(f"[voice_call] using non-bt mic to keep a2dp quality (mic={best})")

    def _maybe_disable_bt_autoswitch(self) -> None:
        """Best-effort disable of WirePlumber bluetooth autoswitch for this session."""
        if not self.bt_disable_autoswitch:
            return
        if self._wpctl_autoswitch_changed:
            return
        pactl = shutil.which("pactl")
        wpctl = shutil.which("wpctl")
        if not pactl or not wpctl:
            return
        sink = _run_capture_text([pactl, "get-default-sink"])
        if not sink or "bluez" not in sink.lower():
            return
        # Only do this if we're *not* using a BT mic (i.e., we intend to keep A2DP).
        if "bluez" in (self.mic_input or "").lower():
            return

        key = "bluetooth.autoswitch-to-headset-profile"
        current = _run_capture_text([wpctl, "settings", key], timeout_s=3.0)
        prev = _parse_boolish(current)
        if prev is None:
            return
        self._wpctl_prev_autoswitch = prev
        if prev is False:
            return  # already disabled
        # Set without --save: temporary for this WirePlumber instance.
        ok = _run_capture_text([wpctl, "settings", key, "false"], timeout_s=3.0)
        if ok is None:
            return
        self._wpctl_autoswitch_changed = True
        print("[voice_call] disabled bt profile autoswitch (wireplumber) for this session")

    def _maybe_restore_bt_autoswitch(self) -> None:
        """Restore WirePlumber bluetooth autoswitch if we changed it."""
        if not self._wpctl_autoswitch_changed:
            return
        wpctl = shutil.which("wpctl")
        if not wpctl:
            return
        prev = self._wpctl_prev_autoswitch
        if prev is None:
            return
        key = "bluetooth.autoswitch-to-headset-profile"
        desired = "true" if prev else "false"
        _ = _run_capture_text([wpctl, "settings", key, desired], timeout_s=3.0)
        self._wpctl_autoswitch_changed = False
        print("[voice_call] restored bt profile autoswitch setting")

    def _vad_threshold(self) -> int:
        # Convert dBFS to 16-bit linear amplitude threshold.
        # Example: -45 dBFS -> ~184.
        return int(32768 * (10 ** (self.mic_vad_db / 20.0)))

    @staticmethod
    def _pcm_rms_16le(frame: bytes) -> int:
        # Avoid numpy dependency; compute RMS in pure Python.
        if not frame:
            return 0
        count = len(frame) // 2
        if count <= 0:
            return 0
        total = 0
        # Little-endian signed 16-bit samples.
        for i in range(0, count * 2, 2):
            sample = int.from_bytes(frame[i : i + 2], byteorder="little", signed=True)
            total += sample * sample
        return int(math.sqrt(total / count)) if total > 0 else 0

    async def _mic_pcm_to_encoder_loop(self) -> None:
        """Read PCM from capture proc, apply VAD/noise gate, write PCM to encoder stdin."""
        cap = self._mic_capture_proc
        enc = self._mic_encode_proc
        if not cap or not enc or not cap.stdout or not enc.stdin:
            return

        bytes_per_sample = 2
        samples_per_frame = int(self.mic_sample_rate * (self.mic_frame_duration_ms / 1000.0))
        frame_bytes = samples_per_frame * bytes_per_sample
        threshold = self._vad_threshold()
        hangover_frames = max(0, int(self.mic_vad_hangover_ms / max(1, self.mic_frame_duration_ms)))
        flush_every_frames = max(1, int(self.config.get("mic_flush_every_frames", 3)))
        frames_since_flush = 0
        loop = asyncio.get_running_loop()
        buf = bytearray()

        try:
            while self.connected and self._mic_uplink_running:
                # If we're injecting a prompt, drain mic bytes but don't write to the encoder.
                # This avoids mixing two PCM sources into a single OGG stream.
                if self._speak_uplink_active:
                    _ = await loop.run_in_executor(None, cap.stdout.read, 4096)
                    await asyncio.sleep(self.mic_frame_duration_ms / 1000.0)
                    continue

                # Duck mic while downlink audio is flowing to avoid feedback loops.
                if self.mic_duck_downlink and (time.time() - self._downlink_last_audio_ts) < (
                    self.mic_duck_hold_ms / 1000.0
                ):
                    # Drain capture stdout so ffmpeg doesn't block on a full pipe.
                    _ = await loop.run_in_executor(None, cap.stdout.read, 4096)
                    # Keep timing by injecting silence frames.
                    await loop.run_in_executor(None, enc.stdin.write, b"\x00" * frame_bytes)
                    frames_since_flush += 1
                    if frames_since_flush >= flush_every_frames:
                        frames_since_flush = 0
                        await loop.run_in_executor(None, enc.stdin.flush)
                    await asyncio.sleep(self.mic_frame_duration_ms / 1000.0)
                    continue

                chunk = await loop.run_in_executor(None, cap.stdout.read, 4096)
                if not chunk:
                    if cap.poll() is not None:
                        break
                    await asyncio.sleep(0.005)
                    continue
                buf.extend(chunk)
                while len(buf) >= frame_bytes and self.connected and self._mic_uplink_running:
                    if self._speak_uplink_active:
                        break
                    frame = bytes(buf[:frame_bytes])
                    del buf[:frame_bytes]

                    if self.mic_vad_enabled:
                        rms = self._pcm_rms_16le(frame)
                        if rms >= threshold:
                            self._vad_open_frames = hangover_frames
                            gated = frame
                        else:
                            if getattr(self, "_vad_open_frames", 0) > 0:
                                self._vad_open_frames -= 1
                                gated = frame
                            else:
                                # Silence suppression: replace low-energy mic frames with clean zero PCM.
                                # This keeps timing stable while preventing ambient noise from being
                                # interpreted as speech.
                                gated = b"\x00" * frame_bytes
                    else:
                        gated = frame

                    if gated:
                        await loop.run_in_executor(None, enc.stdin.write, gated)
                        frames_since_flush += 1
                    if frames_since_flush >= flush_every_frames:
                        frames_since_flush = 0
                        await loop.run_in_executor(None, enc.stdin.flush)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[voice_call] mic pcm loop error: {e}")

    async def _mic_encoder_to_ws_loop(self) -> None:
        """Read OGG/Opus bytes from encoder stdout and send to Moshi as kind=0x01 frames."""
        enc = self._mic_encode_proc
        if not enc or not enc.stdout:
            return
        loop = asyncio.get_running_loop()
        try:
            while self.connected and self._mic_uplink_running and self.ws:
                chunk = await loop.run_in_executor(None, enc.stdout.read, self.mic_chunk_bytes)
                if not chunk:
                    if enc.poll() is not None:
                        break
                    await asyncio.sleep(0.005)
                    continue
                ws = self.ws
                if ws is None:
                    break
                if self._debug_uplink_frames and self._debug_uplink_logged < self._debug_uplink_frames:
                    self._debug_uplink_logged += 1
                    sig = chunk[:4]
                    sig_ascii = ""
                    if sig and all(32 <= b < 127 for b in sig):
                        sig_ascii = sig.decode("ascii", errors="ignore")
                    print(
                        "[voice_call] uplink "
                        f"{self._debug_uplink_logged}/{self._debug_uplink_frames} "
                        f"len={len(chunk)} sig={sig_ascii or sig!r} first16={chunk[:16].hex()}"
                    )
                async with self._send_lock:
                    await ws.send(b"\x01" + chunk)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[voice_call] mic uplink error: {e}")
        finally:
            self._mic_uplink_running = False

    def _start_local_player(self) -> None:
        """Start local audio player process for downlink audio."""
        # If an old process exists (running or exited), ensure it doesn't linger as a zombie.
        if self._player_proc:
            if self._player_proc.poll() is None:
                return
            self._stop_local_player()

        ffplay = shutil.which("ffplay")
        mpv = shutil.which("mpv")
        backend = self.playback_backend

        def build_mpv_cmd() -> list[str] | None:
            if not mpv:
                return None
            # Avoid --really-quiet: it can suppress the errors that explain why playback died.
            return [
                mpv,
                "--no-video",
                "--msg-level=all=warn",
                "--cache=yes",
                f"--cache-secs={self.mpv_cache_secs}",
                f"--demuxer-readahead-secs={self.mpv_readahead_secs}",
                "--demuxer-max-bytes=4M",
                "-",
            ]

        def build_ffplay_cmd() -> list[str] | None:
            if not ffplay:
                return None
            # Moshi sends OGG/Opus pages; ffplay can decode directly from stdin.
            # Use moderate buffering for better audio quality (lower crackle).
            cmd = [
                ffplay,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nodisp",
                "-autoexit",
            ]
            if self.ffplay_low_delay:
                cmd.extend([
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                ])
            cmd.extend([
                "-f",
                "ogg",
                "-i",
                "pipe:0",
            ])
            return cmd

        candidates: list[tuple[str, list[str]]] = []
        if backend in ("auto", "mpv"):
            cmd = build_mpv_cmd()
            if cmd:
                candidates.append(("mpv", cmd))
            cmd = build_ffplay_cmd()
            if cmd:
                candidates.append(("ffplay", cmd))
        elif backend == "ffplay":
            cmd = build_ffplay_cmd()
            if cmd:
                candidates.append(("ffplay", cmd))
            cmd = build_mpv_cmd()
            if cmd:
                candidates.append(("mpv", cmd))
        else:
            cmd = build_mpv_cmd()
            if cmd:
                candidates.append(("mpv", cmd))
            cmd = build_ffplay_cmd()
            if cmd:
                candidates.append(("ffplay", cmd))

        if not candidates:
            print("[voice_call] no local player found (install ffplay or mpv)")
            return

        last_err: object | None = None
        for name, cmd in candidates:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                # Give the player a moment to fail fast (common when the sink/profile flips).
                time.sleep(0.15)
                if proc.poll() is not None:
                    last_err = RuntimeError(f"{name} exited immediately (code={proc.returncode})")
                    self._reap_process(proc)
                    continue

                self._player_proc = proc
                self._player_name = name
                self._player_bytes = 0
                self._playback_restart_count += 1
                self._player_stderr_thread = threading.Thread(
                    target=self._drain_player_stderr,
                    args=(proc,),
                    daemon=True,
                )
                self._player_stderr_thread.start()
                self._playback_frame_counter = 0
                print(f"[voice_call] local playback enabled via {name}")
                return
            except Exception as e:
                last_err = e
                continue

        print(f"[voice_call] failed to start local player: {last_err}")

    def _drain_player_stderr(self, proc: subprocess.Popen) -> None:
        """Surface player diagnostics instead of swallowing silent failures."""
        if not proc.stderr:
            return
        try:
            # Keep logs light: only forward non-empty diagnostic lines.
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"[voice_call] {self._player_name}: {line}")
        except Exception:
            return

    def _write_playback(self, payload: bytes) -> None:
        """Write downlink audio bytes to local player stdin."""
        if not payload or not self.playback_enabled:
            return
        proc = self._player_proc
        if not proc or proc.poll() is not None or proc.stdin is None:
            if self.playback_enabled and self._playback_restart_count < 5:
                self._start_local_player()
                proc = self._player_proc
            if not proc or proc.poll() is not None or proc.stdin is None:
                return
        try:
            with self._player_lock:
                proc.stdin.write(payload)
                self._playback_frame_counter += 1
                if self._playback_frame_counter >= self._playback_flush_every_frames:
                    proc.stdin.flush()
                    self._playback_frame_counter = 0
            self._player_bytes += len(payload)
        except BrokenPipeError:
            self._stop_local_player()
        except Exception:
            # Keep operator alive if playback fails.
            self._stop_local_player()

    def _stop_local_player(self) -> None:
        proc = self._player_proc
        self._player_proc = None
        if not proc:
            return
        self._reap_process(proc)

    @staticmethod
    def _reap_process(proc: subprocess.Popen) -> None:
        """Terminate and wait() on a subprocess so it doesn't linger as a zombie."""
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=0.8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            with contextlib.suppress(Exception):
                proc.wait(timeout=0.8)
