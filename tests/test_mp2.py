import unittest, logging
from pyquante2.geo.molecule import read_xyz
from pyquante2 import rhf, basisset, h2, lih, ch4, h2o, mp2


HBr = read_xyz('./molfiles/HBr.xyz')

H2O4 = read_xyz('./molfiles/H2O_4.xyz')

N8 = read_xyz('./molfiles/N8.xyz')

class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=3e-4):
        x = abs(2*(a-b)/(a+b))
        if x > prec:
            raise AssertionError("%.9f is equal %.9f with precision %.9f)" % (a, b, x))


class test_mp2(unittest.TestCase, PyQuanteAssertions):
    def test_H2(self):
        bfs = basisset(h2,'cc-pvdz')
        solver=rhf(h2,bfs)
        solver.converge()
        nvirt = len(bfs)-h2.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,h2.nocc(),nvirt)
        self.assertPrecisionEqual(emp2, -0.026304104341)

    def test_LiH(self):
        bfs = basisset(lih,'cc-pvdz')
        solver=rhf(lih, bfs, libint=True)
        solver.converge()
        nvirt = len(bfs)-lih.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,lih.nocc(),nvirt)
        self.assertPrecisionEqual(emp2, -0.023948620832)

    def test_H2O(self):
        bfs = basisset(h2o,'cc-pvdz')
        solver=rhf(h2o, bfs, libint=True)
        solver.converge()
        nvirt = len(bfs)-h2o.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,h2o.nocc(),nvirt)
        self.assertPrecisionEqual(emp2, -0.206440187835)

    def test_CH4(self):
        bfs = basisset(ch4,'cc-pvdz')
        solver=rhf(ch4, bfs, libint=True)
        solver.converge()
        nvirt = len(bfs)-ch4.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,ch4.nocc(),nvirt)
        self.assertPrecisionEqual(emp2, -0.166640105042)

    def test_HBr(self):
        bfs = basisset(HBr,'cc-pvdz')
        solver=rhf(HBr, bfs, libint=True)
        solver.converge()
        nvirt = len(bfs)-HBr.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,HBr.nocc(),nvirt)
        self.assertPrecisionEqual(emp2, -0.153284373119)

    def test_N8(self):
        bfs = basisset(N8,'cc-pvdz')
        solver=rhf(N8, bfs, libint=True)
        solver.converge()
        nvirt = len(bfs)-N8.nocc()
        emp2 = mp2(solver.i2,solver.orbs,solver.orbe,N8.nocc(),nvirt)
        self.assertPrecisionEqual(emp2, -1.328348475507)


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
