#!/usr/bin/env python3
"""
Integration test for the data flow fix.
Verifies that updating beta_posteriors triggers a rebuild of the training dataset.
"""

import sys
import shutil
import subprocess
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

class TestDataFlow(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.test_dir = self.root / "tests/tmp_integration"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        # Setup artifacts
        self.run_dir = self.test_dir / "artifacts/test_run"
        self.run_dir.mkdir(parents=True)
        
        # 1. grid.npz
        np.savez(self.run_dir / "grid.npz", x=np.zeros(5), y=np.zeros(5))
        
        # 2. double_well_data.csv (Static)
        df = pd.DataFrame({
            "x": np.zeros(5),
            "y": np.zeros(5),
            "q_hat": np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
            "weight": np.ones(5) * 100.0,
            "n_hi": np.zeros(5), "n_lo": np.zeros(5), # Extra cols
            "dt_hi": np.zeros(5), "dt_lo": np.zeros(5),
            "sigma": np.zeros(5), "t_max": np.zeros(5),
            "var": np.zeros(5)
        })
        # Need double_well_data.csv in repo root for active_round to find it
        # But active_round defaults to repo_root/double_well_data.csv.
        # I should put a dummy one there? No, I shouldn't overwrite project files.
        # active_round hardcodes: static_csv_path=repo_root / "double_well_data.csv"
        # I can't easily override that in active_round.py args without changing code.
        # BUT, rebuild_dataset is called with repo_root / "double_well_data.csv".
        
        # Hack: I'll trust that the real double_well_data.csv exists in the repo root
        # and has matching length? No, grid size might match.
        # The real double_well_data.csv has ~40k points. My dummy has 5.
        # This will fail len check in rebuild_dataset.
        
        # Workaround: Create a dummy double_well_data.csv in self.root just for this test?
        # Risky.
        # Better: create a grid that matches the REAL double_well_data.csv length?
        # Or mock rebuild_dataset? No, testing integration.
        
        # Let's see if I can override the path in active_round.py?
        # No, it's hardcoded in the call to rebuild_dataset.
        pass

    def test_flow(self):
        # Since I can't easily replace double_well_data.csv without editing active_round.py,
        # I will check if I can run rebuild_dataset directly to verify IT works 
        # (which I did in unit test).
        # And check if active_round calls it.
        
        # I'll rely on the fact that I modified active_round.py to call it.
        # If I run active_round.py with --skip-train --skip-fuse, it should still rebuild dataset.
        
        # I will verify that the rebuild happens by checking file modification times / hashes
        # using the REAL double_well_data.csv and REAL beta_posteriors.
        
        # 1. Copy real artifacts to temp dir
        real_artifacts = self.root / "artifacts/latest_bicep"
        shutil.copy(real_artifacts / "grid.npz", self.run_dir / "grid.npz")
        shutil.copy(real_artifacts / "beta_posteriors.npz", self.run_dir / "beta_posteriors.npz")
        shutil.copy(real_artifacts / "enn.npz", self.run_dir / "enn.npz")
        
        # 2. Run active_round (dry run)
        cmd = [
            sys.executable,
            str(self.root / "active_round.py"),
            "--run-dir", str(self.run_dir),
            "--skip-train",
            "--skip-fuse"
        ]
        
        print("Running round 1...")
        subprocess.run(cmd, check=True, cwd=self.root)
        
        merged_path = self.run_dir / "training_data_merged.csv"
        self.assertTrue(merged_path.exists(), "Merged dataset should be created")
        
        # Read Hash 1
        with open(merged_path, "rb") as f:
            hash1 = f.read()
            
        # 3. Modify beta posteriors
        beta = np.load(self.run_dir / "beta_posteriors.npz")
        alpha = beta["alpha"]
        alpha[0] += 1000  # Significant change
        np.savez(self.run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta["beta"])
        
        # 4. Run round 2
        print("Running round 2...")
        subprocess.run(cmd, check=True, cwd=self.root)
        
        # Read Hash 2
        with open(merged_path, "rb") as f:
            hash2 = f.read()
            
        self.assertNotEqual(hash1, hash2, "Dataset should change when posteriors change")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
