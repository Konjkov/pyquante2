import unittest, logging
from pyquante2 import molecule, rhf, uhf, h2, h2o, lih, li, oh, basisset
from pyquante2.ints.integrals import libint_twoe_integrals
from pyquante2.geo.molecule import read_xyz
from pyquante2.scf.iterators import AveragingIterator


class test_scf(unittest.TestCase):
    def test_h2(self):
        bfs = basisset(h2,'sto-3g')
        solver = rhf(h2,bfs)
        ens = solver.converge()
        self.assertAlmostEqual(solver.energy,-1.117099582955609,6)

    def test_h2_631(self):
        bfs = basisset(h2,'6-31gss')
        solver = rhf(h2,bfs)
        ens = solver.converge()
        self.assertAlmostEqual(solver.energy,-1.1313335790123258)

    def test_lih(self):
        bfs = basisset(lih,'sto-3g')
        solver = rhf(lih,bfs)
        ens = solver.converge()
        self.assertAlmostEqual(solver.energy,-7.8607437,6)

    def test_lih_averaging(self):
        bfs = basisset(lih,'sto-3g')
        solver = rhf(lih,bfs)
        ens = solver.converge(AveragingIterator)
        self.assertAlmostEqual(solver.energy,-7.8607375733271088,6)

    def test_h4(self):
        h4 = molecule([
            (1,  0.00000000,     0.00000000,     0.36628549),
            (1,  0.00000000,     0.00000000,    -0.36628549),
            (1,  0.00000000,     4.00000000,     0.36628549),
            (1,  0.00000000,     4.00000000,    -0.36628549),
            ],
                      units='Angstrom')
        bfs = basisset(h4,'sto-3g')
        solver = rhf(h4,bfs)
        ens = solver.converge()
        self.assertAlmostEqual(solver.energy,-2.234185653441159,6)
        # This is not quite equal to 2x the h2 energy, but very close

    def test_h2o(self):
        bfs = basisset(h2o,'sto-3g')
        solver = rhf(h2o,bfs)
        ens = solver.converge()
        self.assertAlmostEqual(solver.energy,-74.959856675848712)

    def test_h2o_averaging(self):
        bfs = basisset(h2o,'sto-3g')
        solver = rhf(h2o,bfs)
        ens = solver.converge(AveragingIterator)
        self.assertAlmostEqual(solver.energy,-74.959847457272502)

    def test_oh(self):
        bfs = basisset(oh,'sto-3g')
        solver = uhf(oh,bfs)
        Es = solver.converge()
        self.assertAlmostEqual(solver.energy,-74.14666861386641,6)

    def test_li(self):
        bfs = basisset(li,'sto-3g')
        solver = uhf(li,bfs)
        Es = solver.converge()
        self.assertAlmostEqual(solver.energy,-7.2301642412807379)


class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=1e-8):
        x = abs(2*(a-b)/(a+b))
        if x > prec:
            raise AssertionError("%.9f is equal %.9f with precision %.9f)" % (a, b, x))


class test_libint_rhf(unittest.TestCase, PyQuanteAssertions):
    """reference energies obtained from NWCHEM 6.5"""
    def test_CH4(self):
        """CH4 symmetry Td"""
        CH4 = molecule([(6,  0.00000000,     0.00000000,     0.00000000),
                        (1,  1.18989170,     1.18989170,     1.18989170),
                        (1, -1.18989170,    -1.18989170,     1.18989170),
                        (1, -1.18989170,     1.18989170,    -1.18989170),
                        (1,  1.18989170,    -1.18989170,    -1.18989170)],
                        units='Bohr',
                        name='CH4')
        bfs = basisset(CH4,'sto-3g')
        solver = rhf(CH4, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -39.726670467839)

    def test_C2H2Cl2(self):
        """C2H2Cl2 symmetry C2H"""
        C2H2Cl2 = read_xyz('./molfiles/C2H2Cl2.xyz')
        bfs = basisset(C2H2Cl2,'sto-3g')
        solver = rhf(C2H2Cl2, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -967.533150327823)

    def test_H2O_4(self):
        """H2O tethramer symmetry S4"""
        H2O4 = read_xyz('./molfiles/H2O_4.xyz')
        bfs = basisset(H2O4,'sto-3g')
        solver = rhf(H2O4, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -299.909789863537)

    def test_BrF5(self):
        """BrF5 symmetry C4v"""
        BrF5 = read_xyz('./molfiles/BrF5.xyz')
        bfs = basisset(BrF5,'sto-3g')
        solver = rhf(BrF5, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -3035.015731331871)

    def test_HBr(self):
        """HBr"""
        HBr = read_xyz('./molfiles/HBr.xyz')
        bfs = basisset(HBr,'sto-3g')
        solver = rhf(HBr, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -2545.887434128302)

    def test_C8H8(self):
        """C8H8"""
        C8H8 = read_xyz('./molfiles/C8H8.xyz')
        bfs = basisset(C8H8,'sto-6g')
        solver = rhf(C8H8, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -306.765545547300)

    def test_N8(self):
        """N8"""
        N8 = read_xyz('./molfiles/N8.xyz')
        bfs = basisset(N8,'cc-pvdz')
        solver = rhf(N8, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -434.992755329296)


class test_unstable(unittest.TestCase, PyQuanteAssertions):
    """Unstable RHF convergence.
       Different NWCHEM energy with and without autosym.
    """
    def test_B12(self):
        """B12 symmetry Ih"""
        B12 = read_xyz('./molfiles/B12.xyz')
        bfs = basisset(B12,'sto-3g')
        solver = rhf(B12, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -290.579419642829)

    def test_CrCO6(self):
        # FAIL
        """Cr(CO)6 symmetry Oh
        Reference: Whitaker, A.; Jeffery, J. W. Acta Cryst. 1967, 23, 977. DOI: 10.1107/S0365110X67004153
        """
        CrCO6 = read_xyz('./molfiles/CrCO6.xyz')
        bfs = basisset(CrCO6,'sto-3g')
        solver = rhf(CrCO6, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge(iterator=AveragingIterator)
        self.assertPrecisionEqual(solver.energy, -1699.539642257497, prec=1e-4)

    def test_C24(self):
        # FAIL
        """C24 symmetry Th"""
        C24 = read_xyz('./molfiles/C24.xyz')
        bfs = basisset(C24,'sto-3g')
        solver = rhf(C24, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -890.071915453874, prec=1e-3)


class test_libint_uhf(unittest.TestCase, PyQuanteAssertions):
    """reference energies obtained from NWCHEM 6.5"""
    def test_CF3(self):
        """CF3 radical"""
        CF3 = read_xyz('./molfiles/CF3.xyz')
        bfs = basisset(CF3,'sto-3g')
        solver = uhf(CF3, bfs, twoe_factory=libint_twoe_integrals)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -331.480688906400)


def runsuite(verbose=True):
    # To use psyco, uncomment this line:
    #import psyco; psyco.full()
    if verbose: verbosity=2
    else: verbosity=1
    # If you want more output, uncomment this line:
    logging.basicConfig(format="%(message)s",level=logging.DEBUG)
    suite1 = unittest.TestLoader().loadTestsFromTestCase(test_scf)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(test_libint_rhf)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(test_unstable)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(test_libint_uhf)
    alltests = unittest.TestSuite([suite1, suite2, suite3, suite4])
    unittest.TextTestRunner(verbosity=verbosity).run(alltests)
    # Running without verbosity is equivalent to replacing the above
    # two lines with the following:
    #unittest.main()


def debugsuite():
    import cProfile,pstats
    cProfile.run('runsuite()','prof')
    prof = pstats.Stats('prof')
    prof.strip_dirs().sort_stats('time').print_stats(15)


if __name__ == '__main__':
    import sys
    if "-d" in sys.argv:
        debugsuite()
    else:
        runsuite()
