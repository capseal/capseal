import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from rebuild_training_data import rebuild_dataset

class TestDatasetBuilder(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_merge_logic(self):
        # 1. Create dummy static CSV (3 points)
        # Pt 0: Boundary (Weight 100, q=0)
        # Pt 1: Boundary (Weight 100, q=1)
        # Pt 2: Prior Data (Weight 10, q=0.5)
        df = pd.DataFrame({
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "q_hat": [0.0, 1.0, 0.5],
            "weight": [100.0, 100.0, 10.0],
            "other_col": [1, 2, 3]  # Should be preserved
        })
        csv_path = self.test_dir / "static.csv"
        df.to_csv(csv_path, index=False)

        # 2. Create dummy beta posteriors (matches 3 points)
        # Pt 0: No new data (1, 1)
        # Pt 1: 10 successes (11, 1) -> delta_alpha=10
        # Pt 2: 10 failures (1, 11) -> delta_beta=10
        alpha = np.array([1.0, 11.0, 1.0])
        beta = np.array([1.0, 1.0, 11.0])
        
        beta_path = self.test_dir / "beta.npz"
        np.savez(beta_path, alpha=alpha, beta=beta)

        # 3. Run rebuild
        out_path = self.test_dir / "merged.csv"
        rebuild_dataset(self.test_dir, csv_path, beta_path, out_path)

        # 4. Verify output
        res = pd.read_csv(out_path)
        
        # Check Pt 0: Unchanged
        self.assertAlmostEqual(res.iloc[0]["q_hat"], 0.0)
        self.assertAlmostEqual(res.iloc[0]["weight"], 100.0)
        
        # Check Pt 1: 100 trials (100 successes) + 10 new successes
        # Total alpha = 100 + 10 = 110
        # Total beta = 0 + 0 = 0
        # Weight = 110
        # q = 1.0
        self.assertAlmostEqual(res.iloc[1]["q_hat"], 1.0)
        self.assertAlmostEqual(res.iloc[1]["weight"], 110.0)
        
        # Check Pt 2: 10 trials (5 succ, 5 fail) + 10 new failures
        # Static: alpha=5, beta=5
        # Dynamic: delta_alpha=0, delta_beta=10
        # Total: alpha=5, beta=15
        # Weight = 20
        # q = 5/20 = 0.25
        self.assertAlmostEqual(res.iloc[2]["q_hat"], 0.25)
        self.assertAlmostEqual(res.iloc[2]["weight"], 20.0)

        # Check column preservation
        self.assertEqual(res.iloc[2]["other_col"], 3)

if __name__ == "__main__":
    unittest.main()
