#!/usr/bin/env python3
import subprocess
import sys
import resource
import os

def run_sanity_check():
    # Set low ulimit to trigger potential IO errors
    # Note: Python needs some FDs for itself, imports, etc.
    # 256 is usually tight but safe for basics.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Original ulimit: {soft}")
    
    try:
        # Try to set a strict limit. 
        # If 256 is too low for system stability, catch it.
        target_limit = 256
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
        print(f"Set ulimit to: {target_limit}")
    except ValueError as e:
        print(f"Could not set ulimit: {e}")
        return

    # Run a trace large enough to generate > 256 chunks
    # Chunk len = 14. 
    # Steps = 8192 -> ~585 chunks.
    # If the code tries to open too many files, or leaks, it will fail.
    
    cmd = [
        sys.executable, "scripts/zk_geom_demo.py", "prove",
        "--steps", "8192",
        "--row-backend", "geom_stc_rust",
        "--row-archive-dir", "out/sanity_check_archive"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    print("Running pipeline with low ulimit...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print("\n[SUCCESS] Pipeline FAILED as expected (Failed Loud).")
        print("Stderr output:")
        print(result.stderr[-500:]) # Print last 500 chars
    else:
        print("\n[FAILURE] Pipeline PASSED? (It should have failed or we didn't hit the limit).")
        print(result.stdout)

    # Cleanup
    if os.path.exists("out/sanity_check_archive"):
        import shutil
        shutil.rmtree("out/sanity_check_archive")

if __name__ == "__main__":
    run_sanity_check()
