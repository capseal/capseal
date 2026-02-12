"""
CapSeal Operator — Telegram Channel

Sends notifications and receives commands via Telegram Bot API.

Features:
  - Text messages with HTML formatting (safe with underscored filenames)
  - Inline keyboard buttons (approve/deny/investigate)
  - Voice note sending (OGG/OPUS format via multipart upload)
  - Long-polling for incoming user messages
  - Callback query handling for button presses

Setup:
  1. Message @BotFather on Telegram, create a bot
  2. Get the bot token
  3. Send a message to your bot to establish a chat
  4. Run: capseal operator --setup telegram
"""

import json
import asyncio
import html
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramChannel:
    def __init__(self, bot_token: str, chat_id: str,
                 voice_notes: bool = False,
                 decision_buttons: bool = True,
                 **kwargs):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.voice_notes = voice_notes
        self.decision_buttons = decision_buttons
        self.last_update_id = 0

    def _api_url(self, method: str) -> str:
        return API_BASE.format(token=self.bot_token, method=method)

    async def _api_call(self, method: str, data: dict) -> Optional[dict]:
        """Make a Telegram Bot API call (async-wrapped sync HTTP)."""
        url = self._api_url(method)
        payload = json.dumps(data).encode("utf-8")

        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
            result = json.loads(response.read())
            return result
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"[telegram] API error {e.code}: {body}")
            return None
        except (URLError, OSError) as e:
            print(f"[telegram] Network error: {e}")
            return None

    async def send_text(self, text: str, buttons: Optional[list] = None):
        """Send a text message, optionally with inline keyboard buttons."""
        safe_text = html.escape(text or "")
        data = {
            "chat_id": self.chat_id,
            "text": safe_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        if buttons and self.decision_buttons:
            keyboard = []
            row = []
            for btn in buttons:
                row.append({
                    "text": btn["text"],
                    "callback_data": btn["callback"],
                })
                if len(row) >= 3:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)

            data["reply_markup"] = json.dumps({
                "inline_keyboard": keyboard,
            })

        result = await self._api_call("sendMessage", data)
        if result and result.get("ok"):
            return True
        return False

    async def send_voice(self, audio_bytes: bytes, caption: str = ""):
        """Send a voice note (OGG/OPUS format) via multipart upload."""
        if not self.voice_notes or not audio_bytes:
            return await self.send_text(f"🔊 {caption}")

        boundary = "----CapSealVoiceBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
            f"{self.chat_id}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="voice"; filename="voice.ogg"\r\n'
            f"Content-Type: audio/ogg\r\n\r\n"
        ).encode() + audio_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="caption"\r\n\r\n'
            f"{html.escape(caption or '')}\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        url = self._api_url("sendVoice")
        req = Request(url, data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }, method="POST")

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: urlopen(req, timeout=30))
            result = json.loads(resp.read())
            return result.get("ok", False)
        except Exception as e:
            print(f"[telegram] Voice upload error: {e}")
            return await self.send_text(f"🔊 {caption}")

    async def poll_messages(self) -> list[str]:
        """
        Long-poll for new messages and callback queries from user.
        Returns list of text commands.
        """
        commands = []

        data = {
            "offset": self.last_update_id + 1,
            "limit": 10,
            "timeout": 0,  # Non-blocking poll
            "allowed_updates": ["message", "callback_query"],
        }

        result = await self._api_call("getUpdates", data)

        if not result or not result.get("ok"):
            return commands

        for update in result.get("result", []):
            update_id = update.get("update_id", 0)
            self.last_update_id = max(self.last_update_id, update_id)

            # Handle text messages
            message = update.get("message", {})
            text = message.get("text", "")
            if text and str(message.get("chat", {}).get("id")) == str(self.chat_id):
                commands.append(text)

            # Handle callback queries (button presses)
            callback = update.get("callback_query")
            if callback:
                cb_data = callback.get("data", "")
                cb_id = callback.get("id")

                # Acknowledge the callback
                if cb_id:
                    await self._api_call("answerCallbackQuery", {
                        "callback_query_id": cb_id,
                        "text": f"Processing: {cb_data}",
                    })

                # Translate callback to command
                command = self._callback_to_command(cb_data)
                if command:
                    commands.append(command)

        return commands

    def _callback_to_command(self, callback_data: str) -> Optional[str]:
        """Translate inline keyboard callback to a text command."""
        # New format: "action:target" (e.g., "approve:src/auth/token.py")
        if ":" in callback_data:
            action, target = callback_data.split(":", 1)
            action_map = {
                "approve": "/approve",
                "deny": "/deny",
                "diff": "/diff",
                "pause": "/pause",
                "resume": "/resume",
                "end": "/end",
                "investigate": "/investigate",
                "pty_accept": "/pty_accept",
                "pty_reject": "/pty_reject",
            }
            cmd = action_map.get(action)
            if cmd and target:
                return f"{cmd} {target}"
            elif cmd:
                return cmd

        # Legacy fallback for old-style callback data
        legacy = {
            "approve_last": "/approve",
            "deny_confirm": "/deny",
            "show_diff": "/diff",
            "resume_session": "/resume",
            "end_session": "/end",
            "investigate": "/investigate",
        }
        return legacy.get(callback_data)

    async def send_message_with_buttons(self, text: str, buttons: list):
        """Convenience method to send text with buttons."""
        return await self.send_text(text, buttons=buttons)
