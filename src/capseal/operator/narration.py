"""Local voice narration for CapSeal Operator.

This intentionally does NOT go through PersonaPlex/Moshi.

Rationale:
  - PersonaPlex/Moshi is speech-to-speech. Feeding it TTS audio and expecting it
    to "repeat" exact text is unreliable by design.
  - For event narration we want deterministic, exact wording with minimal latency.

For now we use `espeak` because it's widely available and has zero Python deps.
If you want higher quality narration later, add optional backends (piper/OpenAI)
behind config without changing call sites.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class NarrationConfig:
    backend: str = "espeak"
    espeak_speed: int = 165
    espeak_pitch: int = 40


class LocalNarrator:
    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.cfg = NarrationConfig(
            backend=str(cfg.get("backend", "espeak")).strip().lower() or "espeak",
            espeak_speed=int(cfg.get("espeak_speed", 165)),
            espeak_pitch=int(cfg.get("espeak_pitch", 40)),
        )

    @property
    def available(self) -> bool:
        if self.cfg.backend == "espeak":
            return bool(shutil.which("espeak"))
        return False

    async def speak(self, text: str) -> bool:
        """Speak text locally. Returns True if it was played."""
        cleaned = (text or "").strip()
        if not cleaned:
            return False

        backend = self.cfg.backend
        if backend == "espeak":
            return await self._speak_espeak(cleaned)
        return False

    async def _speak_espeak(self, text: str) -> bool:
        espeak = shutil.which("espeak")
        if not espeak:
            return False

        # Run in a thread to keep the daemon loop responsive.
        def _run() -> int:
            cp = subprocess.run(
                [
                    espeak,
                    "-s",
                    str(self.cfg.espeak_speed),
                    "-p",
                    str(self.cfg.espeak_pitch),
                    "--stdin",
                ],
                input=text.encode("utf-8", errors="ignore"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return int(cp.returncode)

        rc = await asyncio.to_thread(_run)
        return rc == 0

