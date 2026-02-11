"""CapSeal Operator — real-time notifications for AI agent sessions."""

from .daemon import OperatorDaemon, SessionContext
from .composer import MessageComposer, Message
from .significance import SignificanceFilter
from .config import load_config, DEFAULT_CONFIG
from .intervention import InterventionChannel

__all__ = [
    "OperatorDaemon",
    "SessionContext",
    "MessageComposer",
    "Message",
    "SignificanceFilter",
    "load_config",
    "DEFAULT_CONFIG",
    "InterventionChannel",
]
