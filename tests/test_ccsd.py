import unittest, logging
from pyquante2.geo.molecule import read_xyz
from pyquante2 import rhf, basisset, h2, lih, h2o, ccsd

class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=5e-7):
        x = abs(2*(a-b)/(a+b))
        if x > prec:
            raise AssertionError("%.9f is equal %.9f with precision %.9f)" % (a, b, x))


class test_ccsd(unittest.TestCase, PyQuanteAssertions):
    def test_H2(self):
        bfs = basisset(h2,'cc-pvdz')
        solver=rhf(h2, bfs, libint=True)
        solver.converge()
        H = solver.i1.T + solver.i1.V
        nvirt = len(bfs)-h2.nocc()
        eccsd = ccsd(solver.i2, solver.orbs, solver.orbe, h2.nocc(), nvirt, H)
        self.assertPrecisionEqual(eccsd, -0.034544318453406)

    def test_LiH(self):
        bfs = basisset(lih,'cc-pvdz')
        solver=rhf(lih, bfs, libint=True)
        solver.converge()
        H = solver.i1.T + solver.i1.V
        nvirt = len(bfs)-lih.nocc()
        eccsd = ccsd(solver.i2, solver.orbs, solver.orbe, lih.nocc(), nvirt, H)
        self.assertPrecisionEqual(eccsd, -0.032399770519793)

    def test_H2O(self):
        bfs = basisset(h2o,'cc-pvdz')
        solver=rhf(h2o, bfs, libint=True)
        solver.converge()
        H = solver.i1.T + solver.i1.V
        nvirt = len(bfs)-h2o.nocc()
        eccsd = ccsd(solver.i2, solver.orbs, solver.orbe, h2o.nocc(), nvirt, H)
        self.assertPrecisionEqual(eccsd, -0.215438874234570)


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


if __name__ == '__main__':
    import sys
    if "-d" in sys.argv:
        debugsuite()
    else:
        runsuite()
