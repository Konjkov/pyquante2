import unittest, logging
import tables as tb
from pyquante2 import molecule, rhf, basisset
from pyquante2.geo.molecule import read_xyz
from pyquante2.scf.iterators import AveragingIterator

CH4 = molecule([(6,  0.000000000000,  0.000000000000,  0.000000000000),
                (1,  0.000000000000, -1.697250289185, -1.200137188939),
                (1,  1.697250289426,  0.000000000000,  1.200137188939),
                (1, -1.697250289426, -0.000000000000,  1.200137188939),
                (1, -0.000000000000,  1.697250289185, -1.200137188939)],
                units='Bohr',
                name='CH4')

class PyQuanteAssertions:
    def assertPrecisionEqual(self, a, b, prec=1e-8):
        x = abs(2*(a-b)/(a+b))
        if x > prec:
            raise AssertionError("%.9f is equal %.9f with precision %.9f)" % (a, b, x))


def hp5(filename, a):
    h5file = tb.open_file(filename, mode='w')
    atom = tb.Atom.from_dtype(a.dtype)
    ds = h5file.createCArray(h5file.root, "integrals", atom, a.shape)
    ds[:] = a
    h5file.close()


class test_rhf_energy(unittest.TestCase, PyQuanteAssertions):
    """reference energies obtained from NWCHEM 6.5"""
    def test_CH4(self):
        """CH4 symmetry Td"""
        bfs = basisset(CH4,'sto-3g')
        solver = rhf(CH4,bfs,libint=True)
        ens = solver.converge()
        hp5('Ham.h5', solver.i1.V + solver.i1.T)
        hp5('TwoS.h5', solver.i2._2e_ints)
        self.assertPrecisionEqual(solver.energy, -39.72591203477140)


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
