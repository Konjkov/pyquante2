import unittest, logging
from pyquante2.ints.integrals import libint_twoe_integrals
from pyquante2.geo.molecule import read_xyz
from pyquante2 import rhf, basisset, h2, lih, ch4, h2o, mp2
from pyquante2.scf.iterators import SCFIterator


class test_mp2(unittest.TestCase):
    def test_H2(self):
        bfs = basisset(h2,'cc-pvdz')
        hamiltonian=rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-h2.nocc()
        emp2 = mp2(hamiltonian, iterator.orbs, iterator.orbe, h2.nocc(), nvirt)
        self.assertAlmostEqual(emp2, -0.026304104341, 6)

    def test_LiH(self):
        bfs = basisset(lih,'cc-pvdz')
        hamiltonian=rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-lih.nocc()
        emp2 = mp2(hamiltonian, iterator.orbs, iterator.orbe, lih.nocc(), nvirt)
        self.assertAlmostEqual(emp2, -0.023948620832, 5)

    def test_H2O(self):
        bfs = basisset(h2o,'cc-pvdz')
        hamiltonian=rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-h2o.nocc()
        emp2 = mp2(hamiltonian, iterator.orbs, iterator.orbe, h2o.nocc(), nvirt)
        self.assertAlmostEqual(emp2, -0.206440187835, 6)

    def test_CH4(self):
        bfs = basisset(ch4,'cc-pvdz')
        hamiltonian=rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-ch4.nocc()
        emp2 = mp2(hamiltonian, iterator.orbs, iterator.orbe, ch4.nocc(), nvirt)
        self.assertAlmostEqual(emp2, -0.166640105042, 5)

    def test_HBr(self):
        HBr = read_xyz('./molfiles/HBr.xyz')
        bfs = basisset(HBr,'cc-pvdz')
        hamiltonian=rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-HBr.nocc()
        emp2 = mp2(hamiltonian, iterator.orbs, iterator.orbe, HBr.nocc(), nvirt)
        self.assertAlmostEqual(emp2, -0.153284373119, 6)

    def test_N8(self):
        # 2.8 Gb memory needed
        N8 = read_xyz('./molfiles/N8.xyz')
        bfs = basisset(N8,'cc-pvdz')
        hamiltonian=rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-N8.nocc()
        emp2 = mp2(hamiltonian, iterator.orbs, iterator.orbe, N8.nocc(), nvirt)
        self.assertAlmostEqual(emp2, -1.328348475507, 6)


def runsuite(verbose=True):
    # To use psyco, uncomment this line:
    #import psyco; psyco.full()
    if verbose: verbosity=2
    else: verbosity=1
    # If you want more output, uncomment this line:
    logging.basicConfig(format="%(message)s",level=logging.DEBUG)
    suite = unittest.TestLoader().loadTestsFromTestCase(test_mp2)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
    # Running without verbosity is equivalent to replacing the above
    # two lines with the following:
    #unittest.main()


if __name__ == '__main__':
    import sys
    if "-d" in sys.argv:
        debugsuite()
    else:
        runsuite()
