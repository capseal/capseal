#!/usr/bin/env python3
import time, random, math
import sys

import importlib.util
import os

# Add current directory to path
sys.path.append(".")

# PRE-LOAD the compiled Rust extension from specific path
# This MUST happen before 'bef_zk' is imported to prevent 'prover.py' 
# from finding the 'bef_rust' source directory instead.
so_path = os.path.join(os.getcwd(), "bef_zk", "bef_rust.so")
if os.path.exists(so_path):
    try:
        spec = importlib.util.spec_from_file_location("bef_rust", so_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bef_rust"] = mod
            spec.loader.exec_module(mod)
            print("Successfully pre-loaded bef_rust extension.")
        else:
            print("Failed to create spec for bef_rust.so")
    except Exception as e:
        print(f"Error loading bef_rust.so: {e}")
else:
    print(f"WARNING: {so_path} not found. Rust optimizations unavailable.")

from bef_zk.fri.prover import fri_prove
from bef_zk.fri.config import FRIConfig
from bef_zk.stc.vc import STCVectorCommitment, VCCommitment

MODULUS = (1 << 61) - 1

def random_codeword(n: int):
    return [random.randrange(MODULUS) for _ in range(n)]

def benchmark(n, chunk_len, final_degree, runs=5, num_queries=10):
    # num_rounds such that n / 2^num_rounds ≈ final_degree
    # i.e., num_rounds = log2(n/final_degree)
    num_rounds = int(math.log2(n // final_degree))
    
    # Ensure domain_size is power of two
    domain_size = 1 << int(math.ceil(math.log2(n)))
    if domain_size != n:
        print(f"Adjusting n from {n} to {domain_size} (power of two)")
        n = domain_size

    config = FRIConfig(
        field_modulus=MODULUS,
        domain_size=n,
        max_degree=final_degree,
        num_rounds=num_rounds,
        num_queries=num_queries
    )
    
    vc = STCVectorCommitment(chunk_len=chunk_len)
    
    py_times = []
    rust_times = []

    print(f"\n=== Benchmark Config ===")
    print(f"  Codeword Size : {n} (2^{int(math.log2(n))})")
    print(f"  Chunk Length  : {chunk_len}")
    print(f"  Target Degree : {final_degree}")
    print(f"  Num Rounds    : {num_rounds}")
    print(f"  Num Queries   : {num_queries}")
    print("=" * 30)

    for i in range(runs):
        print(f"Run {i+1}/{runs}...")
        
        # Generate random data
        cw = random_codeword(n)
        query_indices = random.sample(range(n), num_queries)
        
        # --- Python Execution ---
        # Python flow requires us to perform the initial commitment explicitly
        # and pass it to fri_prove.
        t_start = time.time()
        base_commit = vc.commit(cw)
        fri_prove(
            fri_cfg=config, 
            vc=vc, 
            base_evals=cw, 
            base_commitment=base_commit, 
            query_indices=query_indices, 
            use_rust=False
        )
        py_duration = time.time() - t_start
        py_times.append(py_duration)
        
        # --- Rust Execution ---
        # Rust flow in fri_prove(use_rust=True) handles the full commitment stack internally
        # (re-committing the base layer using Rust). 
        # We pass base_commit just to satisfy the signature (it's largely ignored/overwritten by Rust logic).
        t_start = time.time()
        fri_prove(
            fri_cfg=config, 
            vc=vc, 
            base_evals=cw, 
            base_commitment=base_commit, 
            query_indices=query_indices, 
            use_rust=True
        )
        rust_duration = time.time() - t_start
        rust_times.append(rust_duration)
        
        print(f"  Python: {py_duration:.3f}s | Rust: {rust_duration:.3f}s")

    avg_py = sum(py_times) / runs
    avg_rust = sum(rust_times) / runs
    speedup = avg_py / avg_rust if avg_rust > 0 else 0

    print("\n=== Results ===")
    print(f"Python avg : {avg_py:.3f}s (min {min(py_times):.3f}s)")
    print(f"Rust   avg : {avg_rust:.3f}s (min {min(rust_times):.3f}s)")
    print(f"Speed-up   : {speedup:.2f}x\n")

if __name__ == "__main__":
    benchmark(
        n=1 << 18,        # 262k elements
        chunk_len=256,    # Default chunk length
        final_degree=64,
        runs=5
    )
