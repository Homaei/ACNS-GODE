import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spatial.evt_pot import EVTPeaksOverThreshold

class TestEVTExtremes(unittest.TestCase):
    """
    Ensures mathematical bounding distributions correctly translate configurations limits mathematically.
    """
    def setUp(self):
        self.evt = EVTPeaksOverThreshold(W=100, T_roc=20, delta_max=0.5)

    def test_baseline_fallback(self):
        """
        Verifies warm-up periods explicitly restricting standard logic limits logically geometrically.
        """
        errors = np.array([0.1, 0.12, 0.09, 0.15])
        thresh, is_frozen = self.evt.compute_node_threshold(errors, 0.2)
        
        self.assertFalse(is_frozen)
        self.assertGreater(thresh, 0.1)

if __name__ == '__main__':
    unittest.main()
