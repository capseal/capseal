"""Token and secret redaction for CLI output.

This module provides utilities to mask sensitive tokens before they appear
in logs, stderr output, or user-facing messages. This is a P0 security
control - tokens in subprocess output can leak via terminal scrollback,
CI logs, and screen recordings.

Redacted patterns:
- GitHub tokens: ghp_, github_pat_, gho_, ghu_, ghs_, ghr_
- Bearer tokens: Authorization: Bearer xxx
- API keys: sk-..., pk_..., rk_...
- URLs with embedded tokens: https://user:token@host
- Generic secrets: password=, secret=, token=, key=

Usage:
    from capseal.cli.redact import redact_secrets

    safe_output = redact_secrets(subprocess_stderr)
    click.echo(safe_output)
"""
from __future__ import annotations

import re
from typing import Callable

# Compiled patterns for performance (these get called on every subprocess output)
_PATTERNS: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    # GitHub tokens (various prefixes)
    (re.compile(r'\b(ghp_[A-Za-z0-9]{36,})\b'), '[REDACTED:github_pat]'),
    (re.compile(r'\b(github_pat_[A-Za-z0-9_]{22,})\b'), '[REDACTED:github_pat]'),
    (re.compile(r'\b(gho_[A-Za-z0-9]{36,})\b'), '[REDACTED:github_oauth]'),
    (re.compile(r'\b(ghu_[A-Za-z0-9]{36,})\b'), '[REDACTED:github_user]'),
    (re.compile(r'\b(ghs_[A-Za-z0-9]{36,})\b'), '[REDACTED:github_server]'),
    (re.compile(r'\b(ghr_[A-Za-z0-9]{36,})\b'), '[REDACTED:github_refresh]'),

    # OpenAI/Anthropic API keys
    (re.compile(r'\b(sk-[A-Za-z0-9]{20,})\b'), '[REDACTED:api_key]'),
    (re.compile(r'\b(sk-proj-[A-Za-z0-9_-]{20,})\b'), '[REDACTED:api_key]'),
    (re.compile(r'\b(sk-ant-[A-Za-z0-9_-]{20,})\b'), '[REDACTED:anthropic_key]'),

    # Stripe-style keys
    (re.compile(r'\b(pk_live_[A-Za-z0-9]{20,})\b'), '[REDACTED:stripe_pk]'),
    (re.compile(r'\b(sk_live_[A-Za-z0-9]{20,})\b'), '[REDACTED:stripe_sk]'),
    (re.compile(r'\b(rk_live_[A-Za-z0-9]{20,})\b'), '[REDACTED:stripe_rk]'),

    # Bearer tokens in headers
    (re.compile(r'(Bearer\s+)[A-Za-z0-9_.\-]{20,}', re.IGNORECASE), r'\1[REDACTED:bearer]'),
    (re.compile(r'(Authorization:\s*Bearer\s+)[A-Za-z0-9_.\-]{20,}', re.IGNORECASE), r'\1[REDACTED:bearer]'),

    # URLs with embedded credentials (user:pass@host)
    (re.compile(r'(https?://[^:]+:)[^@]+(@[^\s]+)'), r'\1[REDACTED]\2'),

    # Generic key=value patterns (case insensitive, capture key for context)
    (re.compile(r'(password\s*[=:]\s*)[^\s&"\']+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(secret\s*[=:]\s*)[^\s&"\']+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(token\s*[=:]\s*)[^\s&"\']+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(api_key\s*[=:]\s*)[^\s&"\']+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(apikey\s*[=:]\s*)[^\s&"\']+', re.IGNORECASE), r'\1[REDACTED]'),

    # AWS keys
    (re.compile(r'\b(AKIA[A-Z0-9]{16})\b'), '[REDACTED:aws_access_key]'),
    (re.compile(r'(aws_secret_access_key\s*[=:]\s*)[^\s&"\']+', re.IGNORECASE), r'\1[REDACTED]'),

    # Private keys (PEM format - just redact the whole block marker)
    (re.compile(r'-----BEGIN [A-Z ]+ PRIVATE KEY-----'), '[REDACTED:private_key_start]'),
]


def redact_secrets(text: str) -> str:
    """Redact known secret patterns from text.

    This function applies all redaction patterns to sanitize output before
    displaying to users or writing to logs. It's designed to be fast enough
    for real-time subprocess output filtering.

    Args:
        text: Raw text potentially containing secrets

    Returns:
        Text with secrets replaced by [REDACTED:type] markers

    Example:
        >>> redact_secrets("Error: ghp_abc123xyz...")
        'Error: [REDACTED:github_pat]'
    """
    if not text:
        return text

    result = text
    for pattern, replacement in _PATTERNS:
        if callable(replacement):
            result = pattern.sub(replacement, result)
        else:
            result = pattern.sub(replacement, result)

    return result


def redact_env_dict(env: dict[str, str]) -> dict[str, str]:
    """Redact secret values in an environment dictionary.

    Useful for logging what environment variables were passed to a subprocess
    without leaking their actual values.

    Args:
        env: Environment variable dictionary

    Returns:
        Copy with sensitive values redacted
    """
    # Keys that typically contain secrets
    sensitive_keys = {
        'password', 'secret', 'token', 'key', 'api_key', 'apikey',
        'aws_secret', 'private_key', 'credential', 'auth',
        'github_token', 'gh_token', 'openai_api_key', 'anthropic_api_key',
        'greptile_key', 'greptile_api_key', 'gemini_api_key',
    }

    result = {}
    for key, value in env.items():
        key_lower = key.lower()
        # Check if key contains any sensitive substring
        if any(s in key_lower for s in sensitive_keys):
            result[key] = '[REDACTED]'
        else:
            # Still redact the value in case it contains tokens
            result[key] = redact_secrets(value)

    return result


def is_likely_secret(value: str) -> bool:
    """Check if a string looks like it might be a secret.

    Useful for warning users when they might be about to expose a secret.

    Args:
        value: String to check

    Returns:
        True if the string matches common secret patterns
    """
    if not value or len(value) < 10:
        return False

    # Check against our patterns
    for pattern, _ in _PATTERNS:
        if pattern.search(value):
            return True

    return False


__all__ = ['redact_secrets', 'redact_env_dict', 'is_likely_secret']
