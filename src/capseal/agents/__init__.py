"""Agent implementations for CapSeal.

This module provides concrete agent implementations that work with AgentLoop.

Available agents:
- SimpleAgent: A basic LLM-backed agent that adapts based on risk feedback
"""
from __future__ import annotations

from .simple_agent import SimpleAgent

__all__ = ["SimpleAgent"]
