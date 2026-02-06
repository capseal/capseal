#!/usr/bin/env python3
"""ToyToolEnv v1 - A 3-step tool-using microtask environment.

Environment semantics:
    - Secret target: integer 0-9
    - Tools: lookup() returns noisy target, verify(x) returns noisy correctness
    - Hint: "target is near N" with optional distractors and token truncation
    
Determinism:
    - All randomness derived from provided numpy Generator
    - DISTRACTOR_TOKENS is a fixed module-level constant (never randomly generated)
    - Token-based truncation (not character-based)
"""

from __future__ import annotations

from typing import List

import numpy as np


# Module-level constant - NEVER randomly generate token strings
DISTRACTOR_TOKENS: List[str] = [f"tok{i:03d}" for i in range(128)]


class ToyToolEnv:
    """A 3-step tool-using microtask for agent evaluation.
    
    The task is to guess a secret target (0-9). The agent receives:
    - A hint like "target is near N" (with optional noise/distractors)
    - Access to lookup() which returns noisy target
    - Access to verify(x) which returns noisy correctness
    
    Parameters (grid point):
        tool_noise: int 0-3, noise magnitude for lookup()
        verify_flip: float 0-0.2, probability verify() lies
        hint_ambiguity: int 0-3, offset range for hint target
        distractor_count: int 0-6, number of distractor tokens
        memory_tokens: int 16-128, token limit for hint
        rng: numpy Generator for deterministic randomness
    """
    
    def __init__(
        self,
        tool_noise: int,
        verify_flip: float,
        hint_ambiguity: int,
        distractor_count: int,
        memory_tokens: int,
        rng: np.random.Generator,
    ):
        self.tool_noise = tool_noise
        self.verify_flip = verify_flip
        self.hint_ambiguity = hint_ambiguity
        self.distractor_count = distractor_count
        self.memory_tokens = memory_tokens  # Token limit, not characters
        self.rng = rng
        
        # Generate secret target
        self.target = int(self.rng.integers(0, 10))
        
        # Generate hint (must be after target is set)
        self.hint = self._generate_hint()
    
    def _generate_hint(self) -> str:
        """Generate hint string with optional ambiguity and distractors.
        
        Base hint tokens: ["target", "is", "near", "<N>"]
        Then add distractor tokens and truncate to memory_tokens.
        """
        # Generate ambiguous hint target
        if self.hint_ambiguity > 0:
            offset = int(self.rng.integers(-self.hint_ambiguity, self.hint_ambiguity + 1))
        else:
            offset = 0
        hint_target = (self.target + offset) % 10
        
        # Base hint tokens
        tokens: List[str] = ["target", "is", "near", str(hint_target)]
        
        # Add distractor tokens (from deterministic token list)
        for _ in range(self.distractor_count):
            tokens.append(self.rng.choice(DISTRACTOR_TOKENS))
        
        # Truncate to token limit, then join
        tokens = tokens[:self.memory_tokens]
        return " ".join(tokens)
    
    def lookup(self) -> int:
        """Look up the target with optional noise.
        
        Returns:
            Noisy target value (0-9).
        """
        if self.tool_noise > 0:
            noise = int(self.rng.integers(-self.tool_noise, self.tool_noise + 1))
        else:
            noise = 0
        return (self.target + noise) % 10
    
    def verify(self, x: int) -> bool:
        """Verify if x equals the target (with optional flip).
        
        Args:
            x: Value to verify.
            
        Returns:
            True if x == target (possibly flipped by verify_flip).
        """
        result = (x == self.target)
        if self.verify_flip > 0 and self.rng.random() < self.verify_flip:
            result = not result
        return result
    
    def check_answer(self, guess: int) -> bool:
        """Check if the guess is correct (ground truth, no noise).
        
        Args:
            guess: Agent's final guess.
            
        Returns:
            True if guess == target (no noise applied).
        """
        return guess == self.target
