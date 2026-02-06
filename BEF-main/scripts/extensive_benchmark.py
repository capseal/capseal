#!/usr/bin/env python3
import subprocess
import json
import sys
import os

# Powers of 2 from 2^12 to 2^17
STEPS_LIST = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

def run_bench(steps):
    cmd = [
        sys.executable, "scripts/zk_geom_demo.py", "prove",
        "--steps", str(steps),
        "--profile",
        "--row-backend", "geom_stc_rust",
        "--row-archive-dir", "out/bench_archive"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Error running steps={steps}")
        print(result.stderr)
        return None
        
    lines = result.stdout.strip().splitlines()
    # The JSON profile is expected to be the last line
    try:
        data = json.loads(lines[-1])
        return data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for steps={steps}")
        print(result.stdout)
        return None

def fmt_time(s):
    return f"{s:.3f}s"

def fmt_size(b):
    return f"{b / 1024:.1f} KB"

def main():
    print(f"Running extensive benchmark suite (Steps: {STEPS_LIST}) with --rust-stc...")
    print("-" * 90)
    print(f"{'Steps':<10} | {'Proving Total':<15} | {'Row Commit':<15} | {'Verify':<10} | {'Proof Size':<12} | {'Commit/Total %':<15}")
    print("-" * 90)
    
    # Clean up archive before run
    if os.path.exists("out/bench_archive"):
        import shutil
        shutil.rmtree("out/bench_archive")
    
    for steps in STEPS_LIST:
        data = run_bench(steps)
        if data:
            total = data['time_total_sec']
            commit = data['time_row_commit_sec']
            verify = data['time_verify_sec']
            size = data['proof_size_bytes']
            ratio = (commit / total) * 100
            
            print(f"{steps:<10} | {fmt_time(total):<15} | {fmt_time(commit):<15} | {fmt_time(verify):<10} | {fmt_size(size):<12} | {ratio:.1f}%")
        else:
            print(f"{steps:<10} | FAILED")
    print("-" * 90)

if __name__ == "__main__":
    main()
