import unittest
from ..src.cfd_simple import simple
import torch

class SanityTest(unittest.TestCase):
    def sanity_test(self):
        self.assertEqual(1,1)

class test_constructor_positive_path(unittest.TestCase):
    def test_rows(self):
        sim = simple(2,3,0.1)
        self.assertEqual(sim.rows,2, "incorrect initial domain row values")
        self.assertEqual(sim.cols,3, "incorrect initial domain column values")

    def test_u_domain_values(self):
        sim = simple(2,3,0.1)
        exp = torch.zeros(3,3)

        self.assertTrue(torch.equal(sim.u, exp), "incorrect initial u field")
    
    def test_u_domain_dimensions(self):
        sim = simple(2,3,0.1)

        self.assertEqual(sim.u.size(),(3,3),"Incorrect u domain size")

    def test_v_domain_values(self):
        sim = simple(2,3,0.1)
        exp = torch.zeros(2,4)

        self.assertTrue(torch.equal(sim.v, exp), "incorrect initial v field")
    
    def test_v_domain_dimensions(self):
        sim = simple(2,3,0.1)

        self.assertEqual(sim.v.size(),(2,4),"Incorrect v domain size")

    def test_p_domain_values(self):
        sim = simple(2,3,0.1)
        exp = torch.zeros(2,3)

        self.assertTrue(torch.equal(sim.p, exp), "incorrect initial p field")
    
    def test_p_domain_dimensions(self):
        sim = simple(2,3,0.1)

        self.assertEqual(sim.p.size(),(2,3),"Incorrect p domain size")

    def test_timestep(self):
        sim = simple(2,3,0.1)
        self.assertEqual(sim.ts, 0.1, "incorrect simulation timestep")

class test_momentum_solver(unittest.TestCase):
    def test_momentum_gradient(self):
        sim = simple(2,3,0.1)
        sim.momentum_gradient()
        