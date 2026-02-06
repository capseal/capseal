#!/usr/bin/env python3
import subprocess
import json
import sys
import os
import time
import statistics

# Powers of 2 from 2^12 to 2^18
STEPS_LIST = [4096, 16384, 65536, 262144]
NUM_RUNS = 3
OUT_JSONL = "out/bench_results.jsonl"

def run_bench(steps, mode, run_idx):
    # Ensure fresh archive per run to avoid state leakage
    archive_dir = f"out/bench_{mode}_{steps}_{run_idx}"
    if os.path.exists(archive_dir):
        import shutil
        shutil.rmtree(archive_dir)
    os.makedirs(archive_dir, exist_ok=True)

    cmd = [
        sys.executable, "scripts/zk_geom_demo.py", "prove",
        "--steps", str(steps),
        "--profile",
        "--row-archive-dir", archive_dir
    ]
    
    if mode == "rust":
        cmd.extend(["--row-backend", "geom_stc_rust"])
    elif mode == "python":
        cmd.extend(["--row-backend", "geom_stc_fri"])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run and capture output
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        return None
        
    lines = result.stdout.strip().splitlines()
    data = None
    verified = False
    
    # Parse output for profile data and verification status
    for line in lines:
        if "Verifier result: True" in line:
            verified = True
        elif line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                candidate = json.loads(line)
                if "time_total_sec" in candidate:
                    data = candidate
            except json.JSONDecodeError:
                pass

    if data:
        data["verified"] = verified
        if not verified:
            # In strict mode, we might want to discard unverified runs
            # But for reporting, we flag them.
            pass
            
    return data

def fmt_time(s):
    return f"{s:.3f}s"

def fmt_ms(s):
    return f"{s*1000:.2f}ms"

def main():
    print(f"Running Final Robust Benchmark (Steps: {STEPS_LIST}, Runs: {NUM_RUNS})...")
    print("-" * 130)
    print(f"{ 'Steps':<10} | {'Mode':<8} | {'Total (Med)':<12} | {'Commit (Med)':<12} | {'Verify (Med)':<12} | {'Verified':<10} | {'Speedup':<10}")
    print("-" * 130)
    
    results = []
    
    # Warmup GPU if available (by running a small rust job)
    print("Warming up backend...")
    run_bench(4096, "rust", "warmup")

    for steps in STEPS_LIST:
        # Python Baseline
        py_totals = []
        py_commits = []
        py_verifies = []
        py_verified_flags = []
        
        for i in range(NUM_RUNS):
            data = run_bench(steps, "python", i)
            if data:
                py_totals.append(data['time_total_sec'])
                py_commits.append(data['time_row_commit_sec'])
                py_verifies.append(data['time_verify_sec'])
                py_verified_flags.append(data['verified'])
                results.append({"steps": steps, "mode": "python", "run": i, "data": data})
        
        if py_totals:
            py_med_total = statistics.median(py_totals)
            py_med_commit = statistics.median(py_commits)
            py_med_verify = statistics.median(py_verifies)
            all_verified = all(py_verified_flags)
            ver_str = "YES" if all_verified else "NO"
            print(f"{steps:<10} | {'Python':<8} | {fmt_time(py_med_total):<12} | {fmt_time(py_med_commit):<12} | {fmt_ms(py_med_verify):<12} | {ver_str:<10} | {'1.00x':<10}")
        else:
            py_med_total = None
            print(f"{steps:<10} | {'Python':<8} | {'FAILED':<12} | {'--':<12} | {'--':<12} | {'--':<10} | {'--':<10}")

        # Rust+GPU Optimized
        rust_totals = []
        rust_commits = []
        rust_verifies = []
        rust_verified_flags = []
        
        for i in range(NUM_RUNS):
            data = run_bench(steps, "rust", i)
            if data:
                rust_totals.append(data['time_total_sec'])
                rust_commits.append(data['time_row_commit_sec'])
                rust_verifies.append(data['time_verify_sec'])
                rust_verified_flags.append(data['verified'])
                results.append({"steps": steps, "mode": "rust", "run": i, "data": data})

        if rust_totals:
            rust_med_total = statistics.median(rust_totals)
            rust_med_commit = statistics.median(rust_commits)
            rust_med_verify = statistics.median(rust_verifies)
            all_verified = all(rust_verified_flags)
            ver_str = "YES" if all_verified else "NO"
            speedup = f"{py_med_total / rust_med_total:.2f}x" if py_med_total else "??"
            print(f"{steps:<10} | {'Rust':<8} | {fmt_time(rust_med_total):<12} | {fmt_time(rust_med_commit):<12} | {fmt_ms(rust_med_verify):<12} | {ver_str:<10} | {speedup:<10}")
        else:
            print(f"{steps:<10} | {'Rust':<8} | {'FAILED':<12} | {'--':<12} | {'--':<12} | {'--':<10} | {'--':<10}")
        
        print("-" * 130)

    with open(OUT_JSONL, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Raw results saved to {OUT_JSONL}")

if __name__ == "__main__":
    main()
