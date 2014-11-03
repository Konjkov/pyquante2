import unittest, logging
from pyquante2.ints.integrals import libint_twoe_integrals
from pyquante2.geo.molecule import read_xyz
from pyquante2 import rhf, basisset, h2, lih, h2o, ch4, ccsd
from pyquante2.scf.iterators import SCFIterator


class test_ccsd(unittest.TestCase):
    def test_H2(self):
        bfs = basisset(h2,'cc-pvdz')
        hamiltonian = rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-h2.nocc()
        eccsd = ccsd(hamiltonian, iterator.orbs, iterator.orbe, h2.nocc(), nvirt)
        self.assertAlmostEqual(eccsd, -0.034544318453406, 8)

    def test_LiH(self):
        bfs = basisset(lih,'cc-pvdz')
        hamiltonian = rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-lih.nocc()
        eccsd = ccsd(hamiltonian, iterator.orbs, iterator.orbe, lih.nocc(), nvirt)
        self.assertAlmostEqual(eccsd, -0.032399770519793, 7)

    def test_H2O(self):
        bfs = basisset(h2o,'cc-pvdz')
        hamiltonian = rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-h2o.nocc()
        eccsd = ccsd(hamiltonian, iterator.orbs, iterator.orbe, h2o.nocc(), nvirt)
        self.assertAlmostEqual(eccsd, -0.215438874234570, 7)

    def test_CH4(self):
        bfs = basisset(ch4,'cc-pvdz')
        hamiltonian = rhf(bfs, twoe_factory=libint_twoe_integrals)
        iterator = SCFIterator(hamiltonian)
        iterator.converge()
        self.assertTrue(iterator.converged)
        nvirt = len(bfs)-ch4.nocc()
        eccsd = ccsd(hamiltonian, iterator.orbs, iterator.orbe, ch4.nocc(), nvirt)
        self.assertAlmostEqual(eccsd, -0.189626419684193, 7)


def runsuite(verbose=True):
    # To use psyco, uncomment this line:
    #import psyco; psyco.full()
    if verbose: verbosity=2
    else: verbosity=1
    # If you want more output, uncomment this line:
    logging.basicConfig(format="%(message)s",level=logging.DEBUG)
    suite = unittest.TestLoader().loadTestsFromTestCase(test_ccsd)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
    # Running without verbosity is equivalent to replacing the above
    # two lines with the following:
    #unittest.main()


def debugsuite():
    import cProfile,pstats
    cProfile.run('runsuite()','prof')
    prof = pstats.Stats('prof')
    prof.strip_dirs().sort_stats('time').print_stats(50)


if __name__ == '__main__':
    import sys
    if "-d" in sys.argv:
        debugsuite()
    elif "-l" in sys.argv:
        linedebugsuite()
    else:
        runsuite()
