"""
CapSeal Operator — Natural Language Command Parser

Two-tier parsing:
  1. Fast keyword matching (no API key needed)
  2. LLM fallback via Anthropic API (optional, for ambiguous inputs)

Returns command strings like "/approve src/auth.py" or None.
"""

import asyncio
import json
import os
import re
from typing import Optional, Tuple

# Each entry: (pattern, action_name)
# Ordered so more specific patterns match first
PATTERNS: list[Tuple[re.Pattern, str]] = [
    # Approve — explicit commands
    (re.compile(r"\b(approve|lgtm|go ahead)\b", re.I), "approve"),
    # Approve — natural phrases
    (re.compile(r"\blet\b.*\b(through|pass|go)\b", re.I), "approve"),
    (re.compile(r"\bit(?:'?s?|\s+is)\s*(fine|ok|good|safe)\b", re.I), "approve"),
    (re.compile(r"\b(allow|permit|accept)\s*(it|that|this)?\b", re.I), "approve"),
    # Deny
    (re.compile(r"\b(deny|block|reject|don't allow)\b", re.I), "deny"),
    # Pause / Resume
    (re.compile(r"\b(pause|hold|wait|hang on)\b", re.I), "pause"),
    (re.compile(r"\b(resume|continue|unpause|carry on)\b", re.I), "resume"),
    # Status / Info
    (re.compile(r"\b(status|what's happening|how's it going|update me)\b", re.I), "status"),
    (re.compile(r"\b(trust|trust score|how trusted)\b", re.I), "trust"),
    (re.compile(r"\b(files?|what files|which files)\b", re.I), "files"),
    # Session control
    (re.compile(r"\b(end|stop session|kill|terminate|shut down)\b", re.I), "end"),
    (re.compile(r"\b(help|commands|what can you do)\b", re.I), "help"),
]

# Instruct patterns — checked separately because they capture groups
INSTRUCT_PATTERNS = [
    re.compile(r"\btell\s+(?:claude|it|the agent|agent)\s+to\s+(.+)", re.I),
    re.compile(r"\b(?:ask|have|make)\s+(?:claude|it|the agent)\s+(.+)", re.I),
]

FILE_PATTERN = re.compile(r'[\w./\-]+\.\w{1,10}')


def parse_keyword(text: str) -> Optional[dict]:
    """Fast keyword-based command parse. Returns dict with action or None."""
    # Check instruct patterns first (they have capture groups)
    for pattern in INSTRUCT_PATTERNS:
        m = pattern.search(text)
        if m:
            return {"action": "instruct", "instruction": m.group(1).strip()}

    # Check standard command patterns
    for pattern, action in PATTERNS:
        if pattern.search(text):
            result = {"action": action}
            # For approve/deny, try to extract a file target
            if action in ("approve", "deny"):
                file_match = FILE_PATTERN.search(text)
                if file_match:
                    result["target"] = file_match.group()
            return result

    return None


def _dict_to_command(result: dict) -> str:
    """Convert a parsed dict to a command string for the daemon."""
    action = result.get("action", "")
    target = result.get("target", "")
    instruction = result.get("instruction", "")

    if action == "instruct":
        return f"/instruct {instruction}"
    elif target:
        return f"/{action} {target}"
    else:
        return f"/{action}"


async def parse_llm(text: str) -> Optional[str]:
    """Use Anthropic API to parse ambiguous natural language into a command."""
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    system = (
        "You are a command parser for CapSeal Operator, an AI agent monitoring system. "
        "Parse the user's message into exactly one command.\n\n"
        "Valid commands:\n"
        "/approve [file] - approve a pending action\n"
        "/deny [file] - deny/block a pending action\n"
        "/pause - pause the session\n"
        "/resume - resume the session\n"
        "/status - show session status\n"
        "/trust - show trust score\n"
        "/files - list files touched\n"
        "/end - end the session\n"
        "/instruct <text> - send instruction to the agent\n"
        "/help - show help\n"
        "NONE - if the message is not a command\n\n"
        "Reply with ONLY the command string, nothing else."
    )

    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 50,
        "system": system,
        "messages": [{"role": "user", "content": text}],
    }).encode()

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: urlopen(req, timeout=10))
        result = json.loads(resp.read())
        reply = result.get("content", [{}])[0].get("text", "").strip()
        if reply.startswith("/") and reply.upper() != "NONE":
            return reply
    except (URLError, OSError) as e:
        print(f"[nlp] LLM parse error: {e}")

    return None


async def parse(text: str, llm_fallback: bool = True) -> Optional[str]:
    """Parse natural language into a command string. Keyword first, LLM fallback."""
    # Already a slash command — pass through
    if text.startswith("/"):
        return None

    # Try fast keyword matching
    result = parse_keyword(text)
    if result:
        return _dict_to_command(result)

    if not llm_fallback:
        return None

    # Skip LLM for very short or very long messages
    if len(text) < 3 or len(text) > 200:
        return None

    return await parse_llm(text)
