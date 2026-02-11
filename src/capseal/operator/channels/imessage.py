"""
CapSeal Operator — iMessage Channel (macOS only)

Sends notifications via macOS iMessage using AppleScript.
Only works on macOS (sys.platform == "darwin").

Setup:
  Add to operator.json:
    "channels": { "imessage": { "recipient": "+1234567890" } }
"""

import asyncio
import subprocess
import sys
from typing import Optional


class iMessageChannel:
    """Send notifications via macOS iMessage (AppleScript)."""

    def __init__(self, recipient: str, **kwargs):
        self.recipient = recipient  # phone number or Apple ID email
        self._available = sys.platform == "darwin"

    @property
    def available(self) -> bool:
        return self._available

    async def send_text(self, text: str, buttons: Optional[list] = None):
        """Send a text message via iMessage."""
        if not self._available:
            return False

        # Append button options as text (iMessage has no inline buttons)
        if buttons:
            text += "\n\nReply: " + " / ".join(
                btn.get("text", "") if isinstance(btn, dict) else str(btn)
                for btn in buttons
            )

        # Escape for AppleScript
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        recipient_escaped = self.recipient.replace('"', '\\"')

        script = (
            f'tell application "Messages"\n'
            f'  set targetBuddy to buddy "{recipient_escaped}" of service "iMessage"\n'
            f'  send "{escaped}" to targetBuddy\n'
            f'end tell'
        )

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, timeout=10,
                ),
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[imessage] Send error: {e}")
            return False

    async def send_voice(self, audio_bytes: bytes, caption: str = ""):
        """Voice notes not supported via iMessage AppleScript. Falls back to text."""
        return await self.send_text(f"[voice] {caption}")

    async def poll_messages(self) -> list[str]:
        """iMessage doesn't support easy polling. Returns empty."""
        return []
