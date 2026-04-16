import unittest
import torch
import sys
import os

# Limit paths systematically
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logic.physics.mass_balance import MassBalancePredicate
from src.logic.physics.hazen_williams import HazenWilliamsPredicate

class TestPhysicalLogic(unittest.TestCase):
    """
    Evaluates limiting bounds algebraically ensuring strictly enforced topological mathematically.
    """
    def setUp(self):
        self.nodes = [0, 1]
        self.edges = [(0, 1)]
        self.mb = MassBalancePredicate(2, self.nodes, {1: [(0, 1)]}, {0: [(0, 1)]})
        self.hw = HazenWilliamsPredicate(self.edges, {(0, 1): 0.5})

    def test_temperature_bounding(self):
        """
        Verifies temperatures \tau structurally constraints mathematically limits identically bounded [0.01, 1.0].
        """
        t_mb = self.mb.get_temperatures()
        self.assertTrue(torch.all(t_mb >= 0.01))
        self.assertTrue(torch.all(t_mb <= 1.0))
        
        t_hw = self.hw.get_temperatures()
        self.assertTrue(torch.all(t_hw >= 0.01))
        self.assertTrue(torch.all(t_hw <= 1.0))

if __name__ == '__main__':
    unittest.main()
