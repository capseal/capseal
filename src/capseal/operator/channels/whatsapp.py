"""
CapSeal Operator — WhatsApp Business API Channel

Sends notifications via WhatsApp Business Cloud API.
Uses stdlib urllib only (no external dependencies).

Setup:
  1. Create a Meta Business account
  2. Set up WhatsApp Business API
  3. Get phone_number_id and access_token
  4. Add to operator.json channels.whatsapp config
"""

import asyncio
import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


class WhatsAppChannel:
    """Send notifications via WhatsApp Business Cloud API."""

    def __init__(self, phone_number_id: str, access_token: str,
                 recipient: str, **kwargs):
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.recipient = recipient  # E.164 format: +1234567890
        self.api_base = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"

    async def send_text(self, text: str, buttons: Optional[list] = None):
        """Send a text message, optionally with interactive buttons."""
        payload = {
            "messaging_product": "whatsapp",
            "to": self.recipient,
        }

        # WhatsApp interactive buttons (max 3)
        if buttons and len(buttons) <= 3:
            payload["type"] = "interactive"
            payload["interactive"] = {
                "type": "button",
                "body": {"text": text[:1024]},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": btn.get("callback", ""),
                                "title": btn.get("text", "")[:20],
                            },
                        }
                        for btn in buttons[:3]
                    ]
                },
            }
        else:
            payload["type"] = "text"
            payload["text"] = {"body": text[:4096]}

        return await self._api_call(payload)

    async def send_voice(self, audio_bytes: bytes, caption: str = ""):
        """Send voice note. Falls back to text (media upload not yet implemented)."""
        return await self.send_text(f"[voice] {caption}")

    async def poll_messages(self) -> list[str]:
        """WhatsApp uses webhooks, not polling. Returns empty."""
        return []

    async def _api_call(self, payload: dict) -> bool:
        """Make a WhatsApp Business API call."""
        req = Request(
            self.api_base,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
        )

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
            result = json.loads(resp.read())
            return bool(result.get("messages"))
        except (URLError, OSError) as e:
            print(f"[whatsapp] API error: {e}")
            return False
