#!/usr/bin/env python3
import sys, os, random, math
import importlib.util

# Ensure modules are found
sys.path.append(".")

# PRE-LOAD compiled extension
so_path = os.path.join(os.getcwd(), "bef_zk", "bef_rust.so")
if os.path.exists(so_path):
    try:
        spec = importlib.util.spec_from_file_location("bef_rust", so_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bef_rust"] = mod
            spec.loader.exec_module(mod)
        else:
            print("Failed to load extension spec")
    except Exception as e:
        print(f"Error loading bef_rust.so: {e}")
        sys.exit(1)
else:
    print("bef_rust.so not found")
    sys.exit(1)

from bef_zk.fri.prover import fri_prove
from bef_zk.fri.verifier import fri_verify
from bef_zk.fri.config import FRIConfig
from bef_zk.stc.vc import STCVectorCommitment

MODULUS = (1 << 61) - 1

def random_codeword(n: int):
    return [random.randrange(MODULUS) for _ in range(n)]

def test_interop():
    n = 1 << 16  # 65k
    chunk_len = 256
    final_degree = 64
    num_queries = 10
    num_rounds = int(math.log2(n // final_degree))

    config = FRIConfig(
        field_modulus=MODULUS,
        domain_size=n,
        max_degree=final_degree,
        num_rounds=num_rounds,
        num_queries=num_queries
    )

    vc = STCVectorCommitment(chunk_len=chunk_len)
    
    print("Generating random codeword...")
    cw = random_codeword(n)
    
    print("Committing to base layer (Python)...")
    base_commit = vc.commit(cw)
    
    indices = random.sample(range(n), num_queries)
    expected_values = [cw[i] for i in indices]
    
    print("Generating proof (Rust Backend)...")
    try:
        proof = fri_prove(
            fri_cfg=config,
            vc=vc,
            base_evals=cw,
            base_commitment=base_commit,
            query_indices=indices,
            use_rust=True
        )
    except Exception as e:
        print(f"FAILED to generate proof: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Verifying proof (Python Verifier)...")
    try:
        valid = fri_verify(
            fri_cfg=config,
            vc=vc,
            base_commitment=base_commit,
            proof=proof,
            expected_query_points=indices,
            expected_values=expected_values
        )
    except Exception as e:
        print(f"Verification crashed: {e}")
        import traceback
        traceback.print_exc()
        return

    if valid:
        print("SUCCESS: Rust proof verified by Python verifier!")
    else:
        print("FAILURE: Proof rejected by verifier.")

if __name__ == "__main__":
    test_interop()
