# Geom Backend

Purpose: didactic STARK backend for demonstrating instance binding and capsule composition. Not optimized for performance.

What it proves: a small AIR over a 2×2 matrix + counter; public outputs include `final_cnt` for policy checks.

Bindings enforced: row params (root, chunk_len, arity) and `instance_hash` absorption; verifier re‑derives program/vk/AIR/FRI hashes.

Limitations: Python implementation; JSON I/O; non‑native hash for STC; small parameters; redundant commitments (STC + FRI).

