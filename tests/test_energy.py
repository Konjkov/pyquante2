import unittest, logging
from pyquante2 import molecule, rhf, uhf, basisset
from pyquante2.geo.molecule import read_xyz
from pyquante2.scf.iterators import AveragingIterator


CH4 = molecule([(6,  0.00000000,     0.00000000,     0.00000000),
                (1,  1.18989170,     1.18989170,     1.18989170),
                (1, -1.18989170,    -1.18989170,     1.18989170),
                (1, -1.18989170,     1.18989170,    -1.18989170),
                (1,  1.18989170,    -1.18989170,    -1.18989170)],
                units='Bohr',
                name='CH4')

HBr = read_xyz('./molfiles/HBr.xyz')

C2H2Cl2 = read_xyz('./molfiles/C2H2Cl2.xyz')

H2O4 = read_xyz('./molfiles/H2O_4.xyz')

BrF5 = read_xyz('./molfiles/BrF5.xyz')

B12 = read_xyz('./molfiles/B12.xyz')

C24 = read_xyz('./molfiles/C24.xyz')

CrCO6 = read_xyz('./molfiles/CrCO6.xyz')

C8H8 = read_xyz('./molfiles/C8H8.xyz')

N8 = read_xyz('./molfiles/N8.xyz')

CF3 = read_xyz('./molfiles/CF3.xyz')

class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=2e-8):
        x = abs(2*(a-b)/(a+b))
        if x > prec:
            raise AssertionError("%.9f is equal %.9f with precision %.9f)" % (a, b, x))


class test_rhf_energy(unittest.TestCase, PyQuanteAssertions):
    """reference energies obtained from NWCHEM 6.5"""
    def test_CH4_solver(self):
        """CH4 symmetry Td"""
        bfs = basisset(CH4,'sto-3g')
        solver = rhf(CH4,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -39.726670467839)

    def test_C2H2Cl2_solver(self):
        """C2H2Cl2 symmetry C2H"""
        bfs = basisset(C2H2Cl2,'sto-3g')
        solver = rhf(C2H2Cl2,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -967.533150327823)

    def test_H2O_4_simple(self):
        """H2O tethramer symmetry S4"""
        bfs = basisset(H2O4,'sto-3g')
        solver = rhf(H2O4,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -299.909789863537)

    def test_BrF5_simple(self):
        """BrF5 symmetry C4v"""
        bfs = basisset(BrF5,'sto-3g')
        solver = rhf(BrF5,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -3035.015731331871)

    def test_HBr_simple(self):
        """HBr"""
        bfs = basisset(HBr,'sto-3g')
        solver = rhf(HBr,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -2545.887434128302)

    def test_C8H8_simple(self):
        """C8H8"""
        bfs = basisset(C8H8,'sto-6g')
        solver = rhf(C8H8,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -306.765545547300)

    def test_N8_simple(self):
        """N8"""
        bfs = basisset(N8,'cc-pvdz')
        solver = rhf(N8,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -434.992755329296)


class test_unstable(unittest.TestCase, PyQuanteAssertions):
    """Unstable RHF convergence.
       Different NWCHEM energy with and without autosym.
    """
    def test_B12_solver(self):
        """B12 symmetry Ih"""
        bfs = basisset(B12,'sto-3g')
        solver = rhf(B12,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -290.579419642829)

    def test_CrCO6_simple(self):
        # FAIL
        """Cr(CO)6 symmetry Oh
        Reference: Whitaker, A.; Jeffery, J. W. Acta Cryst. 1967, 23, 977. DOI: 10.1107/S0365110X67004153
        """
        bfs = basisset(CrCO6,'sto-3g')
        solver = rhf(CrCO6,bfs,libint=True)
        ens = solver.converge(iterator=AveragingIterator)
        self.assertPrecisionEqual(solver.energy, -1699.539642257497, prec=1e-4)

    def test_C24_simple(self):
        # FAIL
        """C24 symmetry Th"""
        bfs = basisset(C24,'sto-3g')
        solver = rhf(C24,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -890.071915453874, prec=1e-3)


class test_uhf_energy(unittest.TestCase, PyQuanteAssertions):
    """reference energies obtained from NWCHEM 6.5"""
    def test_CF3_solver(self):
        """CF3 radical"""
        bfs = basisset(CF3,'sto-3g')
        solver = uhf(CF3,bfs,libint=True)
        ens = solver.converge()
        print ens
        self.assertPrecisionEqual(solver.energy, -331.480688906400, prec=4e-8)


def runsuite(verbose=True):
    # To use psyco, uncomment this line:
    #import psyco; psyco.full()
    if verbose: verbosity=2
    else: verbosity=1
    # If you want more output, uncomment this line:
    logging.basicConfig(format="%(message)s",level=logging.DEBUG)
    suite = unittest.TestLoader().loadTestsFromTestCase(test_uhf_energy)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
    # Running without verbosity is equivalent to replacing the above
    # two lines with the following:
    #unittest.main()
    return


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
