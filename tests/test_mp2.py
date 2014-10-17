import unittest, logging
from pyquante2.geo.molecule import read_xyz
from pyquante2 import rhf,basisset,h2,mp2


HBr = read_xyz('./molfiles/HBr.xyz')


class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=2e-5):
        x = abs(2*(a-b)/(a+b))
        if x > prec:
            raise AssertionError("%.9f is equal %.9f with precision %.9f)" % (a, b, x))


class test_mp2(unittest.TestCase, PyQuanteAssertions):
    def test_H2(self):
        bfs = basisset(h2,'cc-pvdz')
        solver=rhf(h2,bfs)
        solver.converge()
        nvirt = len(bfs)-h2.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,h2.nocc(),len(bfs)-h2.nocc())
        self.assertPrecisionEqual(emp2, -0.026304104341)

    def test_HBr(self):
        bfs = basisset(HBr,'cc-pvdz')
        solver=rhf(HBr,bfs,libint=True)
        solver.converge()
        nvirt = len(bfs)-h2.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,h2.nocc(),len(bfs)-h2.nocc())
        self.assertPrecisionEqual(emp2, -0.1532843)


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
