#!/usr/bin/env python3
"""ToyAgent v1 - A deterministic agent for the ToyToolEnv.

Agent policy:
    1. Parse hint for target estimate
    2. With P_USE_TOOL probability, also use lookup() and average
    3. With P_VERIFY probability, verify candidate and adjust if wrong
    
Determinism:
    - Uses token-based parsing (not substring index) for hint parsing
    - Explicit integer averaging (not cyclic group mean)
    - All randomness from provided numpy Generator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent_bench.env_toy_v1 import ToyToolEnv


class ToyAgent:
    """A toy agent with fixed policy parameters.
    
    Policy:
        - P_USE_TOOL = 0.7: Probability of using lookup() tool
        - P_VERIFY = 0.5: Probability of verifying candidate
        
    Averaging:
        When using lookup(), computes explicit integer mean:
        candidate = int(round(0.5 * (hint_val + lookup_val))) % 10
        
        NOT cyclic group mean - this is intentional for simplicity.
    """
    
    # Fixed policy parameters
    P_USE_TOOL = 0.7
    P_VERIFY = 0.5
    
    def act(self, env: "ToyToolEnv", rng: np.random.Generator) -> int:
        """Execute the agent's policy on the environment.
        
        Args:
            env: The ToyToolEnv instance.
            rng: numpy Generator for agent's randomness.
            
        Returns:
            Agent's final guess (0-9).
        """
        # Parse hint for initial estimate
        hint_val = self._parse_hint(env.hint)
        
        # Possibly use lookup tool
        if rng.random() < self.P_USE_TOOL:
            lookup_val = env.lookup()
            # Explicit integer mean, not cyclic group mean
            candidate = int(round(0.5 * (hint_val + lookup_val))) % 10
        else:
            candidate = hint_val
        
        # Possibly verify and adjust
        if rng.random() < self.P_VERIFY:
            if not env.verify(candidate):
                candidate = (candidate + 1) % 10
        
        return candidate
    
    def _parse_hint(self, hint: str) -> int:
        """Extract integer after 'near' token using tokenization.
        
        Uses token-based parsing (split on spaces), NOT substring index.
        This prevents matching substrings like "snearly".
        
        Args:
            hint: The hint string from the environment.
            
        Returns:
            Parsed integer value (0-9), defaults to 0 if parsing fails.
        """
        tokens = hint.split()
        try:
            near_idx = tokens.index("near")
            if near_idx + 1 < len(tokens):
                return int(tokens[near_idx + 1]) % 10
        except (ValueError, IndexError):
            pass
        return 0
