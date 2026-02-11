"""
CapSeal Operator — Live Full-Duplex Voice Channel

Manages a persistent WebSocket connection to PersonaPlex for
real-time speech-to-speech interaction during critical events.

Requires: pip install websockets (optional dependency)
"""

import asyncio
import json
from typing import Optional, Callable, Awaitable

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class VoiceCallManager:
    """Manages a live WebSocket voice channel to PersonaPlex."""

    def __init__(self, config: dict):
        self.config = config
        self.ws_url = config.get("personaplex_ws_url", "wss://api.personaplex.io/v1/stream")
        self.voice_preset = config.get("voice_preset", "NATM1")
        self.ws = None
        self.connected = False
        self._on_transcript: Optional[Callable[[str], Awaitable[None]]] = None

    @property
    def available(self) -> bool:
        return HAS_WEBSOCKETS

    async def connect(self) -> bool:
        """Open WebSocket connection to PersonaPlex."""
        if not HAS_WEBSOCKETS:
            print("[voice_call] websockets not installed. pip install websockets")
            return False

        try:
            self.ws = await websockets.connect(
                f"{self.ws_url}?preset={self.voice_preset}",
                ping_interval=20,
            )
            # Send initial config
            await self.ws.send(json.dumps({
                "type": "config",
                "voice_preset": self.voice_preset,
            }))
            self.connected = True
            print("[voice_call] Connected to PersonaPlex")
            return True
        except Exception as e:
            print(f"[voice_call] Connection failed: {e}")
            return False

    async def speak(self, text: str):
        """Send text to be spoken aloud on the live channel."""
        if not self.connected or not self.ws:
            return
        try:
            await self.ws.send(json.dumps({
                "type": "speak",
                "text": text,
            }))
        except Exception as e:
            print(f"[voice_call] Speak error: {e}")
            self.connected = False

    async def listen_loop(self):
        """Background loop that receives transcripts from the call."""
        if not self.ws:
            return
        try:
            async for message in self.ws:
                data = json.loads(message)
                if data.get("type") == "transcript" and self._on_transcript:
                    await self._on_transcript(data.get("text", ""))
        except Exception:
            self.connected = False

    def on_transcript(self, callback: Callable[[str], Awaitable[None]]):
        """Register a callback for incoming voice transcripts."""
        self._on_transcript = callback

    async def disconnect(self):
        """Close the WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.connected = False
