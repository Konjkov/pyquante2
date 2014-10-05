import unittest, logging
import numpy as np
from pyquante2 import molecule, rhf, uhf, basisset
from pyquante2.utils import trace2, geigh, dmat
from pyquante2.ints.integrals import onee_integrals, libint_twoe_integrals


HBr = molecule([( 1,  0.00000000,     0.00000000,     0.00000000),
                (35,  0.00000000,     0.00000000,     1.00000000)],
                units='Angstrom',
                name='HBr')

CH4 = molecule([(6,  0.00000000,     0.00000000,     0.00000000),
                (1,  1.18989170,     1.18989170,     1.18989170),
                (1, -1.18989170,    -1.18989170,     1.18989170),
                (1, -1.18989170,     1.18989170,    -1.18989170),
                (1,  1.18989170,    -1.18989170,    -1.18989170)],
                units='Bohr',
                name='CH4')

C2H2Cl2 = molecule([( 6,  0,    0,  0.5),
                    ( 6,  0,    0, -0.5),
                    ( 1,  0, -0.4, -1.0),
                    ( 1,  0,  0.4,  1.0),
                    (17,  0, -0.4,  1.0),
                    (17,  0,  0.4, -1.0)],
                    units='Angstrom',
                    name='C2H2Cl2')

H2O4 = molecule([(8,  -1.367062,       1.364510,       0.007273),
                 (8,  -1.364510,      -1.367062,      -0.007273),
                 (8,   1.364510,       1.367062,      -0.007273),
                 (8,   1.367062,      -1.364510,       0.007273),
                 (1,  -0.395152,       1.503429,      -0.005375),
                 (1,  -1.503429,      -0.395152,       0.005375),
                 (1,   1.503429,       0.395152,       0.005375),
                 (1,   0.395152,      -1.503429,      -0.005375),
                 (1,  -1.687281,       1.875361,       0.755434),
                 (1,  -1.875361,      -1.687281,      -0.755434),
                 (1,   1.875361,       1.687281,      -0.755434),
                 (1,   1.687281,      -1.875361,       0.755434)],
                 units='Angstrom',
                 name='H2O4')

BrF5 = molecule([(35,  0.0000,    0.0000,   -0.4183),
                 ( 9, -1.2168,   -1.2168,   -0.2169),
                 ( 9,  1.2168,   -1.2168,   -0.2169),
                 ( 9,  0.0000,    0.0000,    1.2858),
                 ( 9,  1.2168,    1.2168,   -0.2169),
                 ( 9, -1.2168,    1.2168,   -0.2169)],
                 units='Angstrom',
                 name='BrF5')

B12 = molecule([(5,   0.00000000,     0.00000000,     2.00000000),
                (5,   1.44721360,    -1.05146222,     0.89442719),
                (5,   1.78885438,     0.00000000,    -0.89442719),
                (5,   0.55278640,     1.70130162,    -0.89442719),
                (5,  -1.44721360,    -1.05146222,    -0.89442719),
                (5,  -1.44721360,     1.05146222,    -0.89442719),
                (5,  -0.55278640,    -1.70130162,     0.89442719),
                (5,   0.00000000,     0.00000000,    -2.00000000),
                (5,  -0.55278640,     1.70130162,     0.89442719),
                (5,   1.44721360,     1.05146222,     0.89442719),
                (5,  -1.78885438,     0.00000000,     0.89442719),
                (5,   0.55278640,    -1.70130162,    -0.89442719)],
                units='Angstrom',
                name='B12')

C24 = molecule([(6,   2.00000000,     3.00000000,     4.00000000),
                (6,  -2.00000000,    -3.00000000,     4.00000000),
                (6,  -2.00000000,     3.00000000,    -4.00000000),
                (6,   2.00000000,    -3.00000000,    -4.00000000),
                (6,   4.00000000,     2.00000000,     3.00000000),
                (6,   4.00000000,    -2.00000000,    -3.00000000),
                (6,  -4.00000000,    -2.00000000,     3.00000000),
                (6,  -4.00000000,     2.00000000,    -3.00000000),
                (6,   3.00000000,     4.00000000,     2.00000000),
                (6,  -3.00000000,     4.00000000,    -2.00000000),
                (6,   3.00000000,    -4.00000000,    -2.00000000),
                (6,  -3.00000000,    -4.00000000,     2.00000000),
                (6,  -2.00000000,    -3.00000000,    -4.00000000),
                (6,   2.00000000,     3.00000000,    -4.00000000),
                (6,   2.00000000,    -3.00000000,     4.00000000),
                (6,  -2.00000000,     3.00000000,     4.00000000),
                (6,  -4.00000000,    -2.00000000,    -3.00000000),
                (6,  -4.00000000,     2.00000000,     3.00000000),
                (6,   4.00000000,     2.00000000,    -3.00000000),
                (6,   4.00000000,    -2.00000000,     3.00000000),
                (6,  -3.00000000,    -4.00000000,    -2.00000000),
                (6,   3.00000000,    -4.00000000,     2.00000000),
                (6,  -3.00000000,     4.00000000,     2.00000000),
                (6,   3.00000000,     4.00000000,    -2.00000000)],
                units='Angstrom',
                name='C24')

CrCO6 = molecule([(24,    0.00000000,     0.00000000,     0.00000000),
                  ( 6,    0.00000000,    -1.90950000,     0.00000000),
                  ( 6,    0.00000000,     1.90950000,     0.00000000),
                  ( 6,   -1.90950000,     0.00000000,     0.00000000),
                  ( 6,    0.00000000,     0.00000000,    -1.90950000),
                  ( 8,    0.00000000,    -3.04600000,     0.00000000),
                  ( 8,    0.00000000,     3.04600000,     0.00000000),
                  ( 8,   -3.04600000,     0.00000000,     0.00000000),
                  ( 8,    0.00000000,     0.00000000,    -3.04600000),
                  ( 6,    0.00000000,     0.00000000,     1.90950000),
                  ( 6,    1.90950000,     0.00000000,     0.00000000),
                  ( 8,    0.00000000,     0.00000000,     3.04600000),
                  ( 8,    3.04600000,     0.00000000,     0.00000000)],
                  units='Angstrom',
                  name='CrCO6')

C8H8 = molecule([(1,   1.40173963,     1.40173963,     1.40173963),
                 (6,   0.77867761,     0.77867761,     0.77867761),
                 (1,   1.40173963,     1.40173963,    -1.40173963),
                 (6,   0.77867761,     0.77867761,    -0.77867761),
                 (1,   1.40173963,    -1.40173963,     1.40173963),
                 (6,   0.77867761,    -0.77867761,     0.77867761),
                 (1,  -1.40173963,     1.40173963,     1.40173963),
                 (6,  -0.77867761,     0.77867761,     0.77867761),
                 (1,   1.40173963,    -1.40173963,    -1.40173963),
                 (6,   0.77867761,    -0.77867761,    -0.77867761),
                 (1,  -1.40173963,     1.40173963,    -1.40173963),
                 (6,  -0.77867761,     0.77867761,    -0.77867761),
                 (1,  -1.40173963,    -1.40173963,     1.40173963),
                 (6,  -0.77867761,    -0.77867761,     0.77867761),
                 (1,  -1.40173963,    -1.40173963,    -1.40173963),
                 (6,  -0.77867761,    -0.77867761,    -0.77867761)],
                 units='Angstrom',
                 name='Cubane')

N8 = molecule([(7,   0.73,     0.73,     0.73),
               (7,   0.73,     0.73,    -0.73),
               (7,   0.73,    -0.73,     0.73),
               (7,  -0.73,     0.73,     0.73),
               (7,   0.73,    -0.73,    -0.73),
               (7,  -0.73,     0.73,    -0.73),
               (7,  -0.73,    -0.73,     0.73),
               (7,  -0.73,    -0.73,    -0.73)],
               units='Angstrom',
               name='N-Cubane')


def scf_simple(geo,basisname='sto-3g',maxiter=25,verbose=False):
    bfs = basisset(geo,basisname)
    i1 = onee_integrals(bfs,geo)
    i2 = libint_twoe_integrals(bfs)
    if verbose: print ("S=\n%s" % i1.S)
    h = i1.T + i1.V
    if verbose: print ("h=\n%s" % h)
    if verbose: print ("T=\n%s" % i1.T)
    if verbose: print ("V=\n%s" % i1.V)
    E,U = geigh(h,i1.S)
    if verbose: print ("E=\n%s" % E)
    if verbose: print ("U=\n%s" % U)
    Enuke = geo.nuclear_repulsion()
    nocc = geo.nocc()
    Eold = Energy = 0
    if verbose: print ("2e ints\n%s" % i2)
    for i in xrange(maxiter):
        D = dmat(U,nocc)
        if verbose: print ("D=\n%s" % D)
        Eone = 2*trace2(h,D)
        G = i2.get_2jk(D)
        Etwo = trace2(D,G)
        H = h+G
        E,U = geigh(H,i1.S)
        Energy = Enuke+Eone+Etwo
        print ("HF: %02d   %10.6f : %10.6f %10.6f %10.6f" % ((i+1),Energy,Enuke,Eone,Etwo))
        if np.isclose(Energy,Eold):
            break
        Eold = Energy
    else:
        print ("Warning: Maxiter %d hit in scf_simple" % maxiter)
    return Energy,E,U


class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=2e-6):
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

    def test_C2H2Cl2_simple(self):
        """C2H2Cl2 symmetry C2H"""
        self.assertPrecisionEqual(scf_simple(C2H2Cl2)[0], -967.533150327823)

    def test_C2H2Cl2_solver(self):
        """C2H2Cl2 symmetry C2H"""
        bfs = basisset(C2H2Cl2,'sto-3g')
        solver = rhf(C2H2Cl2,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -967.533150327823)

    def test_H2O_4_simple(self):
        """H2O tethramer symmetry S4"""
        self.assertPrecisionEqual(scf_simple(H2O4)[0], -299.909789863537)

    def test_BrF5_simple(self):
        """BrF5 symmetry C4v"""
        self.assertPrecisionEqual(scf_simple(BrF5)[0], -3035.015731331871)

    def test_HBr_simple(self):
        """HBr"""
        self.assertPrecisionEqual(scf_simple(HBr)[0], -2545.887434128302)

    def test_C8H8_simple(self):
        """C8H8"""
        self.assertPrecisionEqual(scf_simple(C8H8, basisname='sto-6g')[0], -306.765545547300)

    def test_N8_simple(self):
        """N8"""
        self.assertPrecisionEqual(scf_simple(N8, basisname='cc-pvdz')[0], -434.992755329296 prec=4e-6)

    def test_CrCO6_simple(self):
        """Cr(CO)6 symmetry Oh
        Reference: Whitaker, A.; Jeffery, J. W. Acta Cryst. 1967, 23, 977. DOI: 10.1107/S0365110X67004153
        """
        bfs = basisset(CrCO6,'sto-3g')
        self.assertPrecisionEqual(scf_simple(CrCO6)[0], -1699.539642257497)

    def test_C24_simple(self):
        """C24 symmetry Th"""
        self.assertPrecisionEqual(scf_simple(C24)[0], -890.071915453874)


class test_unstable(unittest.TestCase, PyQuanteAssertions):
    """Unstable RHF convergence.
       Different NWCHEM energy with and without autosym.
    """
    def test_B12_solver(self):
        """B12 symmetry Ih"""
        bfs = basisset(B12,'sto-3g')
        solver = rhf(B12,bfs,libint=True)
        ens = solver.converge()
        self.assertPrecisionEqual(solver.energy, -290.579419642829, prec=1.0)

    def test_B12_simple(self):
        """B12 symmetry Ih"""
        self.assertPrecisionEqual(scf_simple(B12)[0], -290.579419642829, prec=1.0)


class test_profile(unittest.TestCase, PyQuanteAssertions):
    def test_BrF5_simple(self):
        """BrF5 symmetry C4v"""
        import pstats, cProfile
        cProfile.run('scf_simple(BrF5)')

    def test_C8H8_simple(self):
        """C8H8"""
        import pstats, cProfile
        cProfile.run('scf_simple(C8H8, basisname=\'sto-6g\')')

    def test_N8_simple(self):
        """N8"""
        import pstats, cProfile
        cProfile.run('scf_simple(N8, basisname=\'cc-pvdz\')')


def runsuite(verbose=True):
    # To use psyco, uncomment this line:
    #import psyco; psyco.full()
    if verbose: verbosity=2
    else: verbosity=1
    # If you want more output, uncomment this line:
    logging.basicConfig(format="%(message)s",level=logging.DEBUG)
    suite = unittest.TestLoader().loadTestsFromTestCase(test_rhf_energy)
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
