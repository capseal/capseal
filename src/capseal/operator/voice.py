"""
CapSeal Operator — Voice Synthesis

Generates OGG/OPUS audio from text using TTS providers.
Supports OpenAI TTS API and PersonaPlex, both via stdlib urllib.
"""

import asyncio
import json
import os
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


class VoiceSynthesizer:
    """Synthesize speech from text for voice note notifications."""

    def __init__(self, config: dict):
        self.provider = config.get("provider", "openai")
        self.openai_voice = config.get("openai_voice", "onyx")
        self.personaplex_url = config.get("personaplex_url", "https://api.personaplex.io/v1")
        self.voice_preset = config.get("voice_preset", "NATM1")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech, return OGG/OPUS bytes or None on failure."""
        if self.provider == "openai":
            return await self._openai_tts(text)
        elif self.provider == "personaplex":
            return await self._personaplex_tts(text)
        return None

    async def _openai_tts(self, text: str) -> Optional[bytes]:
        """Call OpenAI TTS API. Returns OPUS audio bytes."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[voice] OPENAI_API_KEY not set, skipping TTS")
            return None

        payload = json.dumps({
            "model": "tts-1",
            "voice": self.openai_voice,
            "input": text[:4096],
            "response_format": "opus",
        }).encode()

        req = Request(
            "https://api.openai.com/v1/audio/speech",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: urlopen(req, timeout=15))
            return resp.read()
        except (URLError, OSError) as e:
            print(f"[voice] OpenAI TTS error: {e}")
            return None

    async def _personaplex_tts(self, text: str) -> Optional[bytes]:
        """Call PersonaPlex TTS endpoint. Returns OGG audio bytes."""
        hf_token = os.environ.get("HF_TOKEN")

        payload = json.dumps({
            "text": text[:4096],
            "preset": self.voice_preset,
            "format": "ogg",
        }).encode()

        headers = {"Content-Type": "application/json"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        req = Request(
            f"{self.personaplex_url}/synthesize",
            data=payload,
            headers=headers,
        )

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: urlopen(req, timeout=20))
            return resp.read()
        except (URLError, OSError) as e:
            print(f"[voice] PersonaPlex TTS error: {e}")
            return None
