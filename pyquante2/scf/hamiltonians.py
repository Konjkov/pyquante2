from pyquante2.utils import dmat, trace2, geigh
from pyquante2.grid.grid import grid
from pyquante2.ints.integrals import onee_integrals, twoe_integrals


class hamiltonian(object):
    """ Class that represent hamiltonian manipulation methods, not state
    """
    name = 'abstract'
    def __init__(self, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        self.geo = bfs.atoms
        self.bfs = bfs
        self.Enuc = self.geo.nuclear_repulsion()
        self.i1 = onee_factory(bfs, bfs.atoms)
        self.i2 = twoe_factory(bfs)
        self.h = self.i1.T + self.i1.V


class rhf(hamiltonian):
    """ RHF hamiltonian
    """
    name = 'RHF'

    def fock(self, D):
        """ Fock matrix
        """
        h = self.h
        G = self.i2.get_2jk(D)
        return h + G

    def eigenv(self, F):
        """ Eigenvalues and Eigenvectors
        """
        orbe, orbs = geigh(F, self.i1.S)
        return orbe, orbs

    def density(self, orbs):
        """ Electron density
        """
        return dmat(orbs, self.geo.nocc())

    def energy(self, D, F):
        """ Total energy
        """
        h = self.h
        G = F - h # TODO: not optimal
        Eone = 2 * trace2(h, D)
        Etwo = trace2(D, G)
        return self.Enuc + Eone + Etwo


class rdft(rhf):
    """Hamiltonian for DFT calculations. Adds a grid to RHF iterator."""
    name = 'DFT'

    def __init__(self, geo, bfs):
        rhf.__init__(self, geo, bfs)
        self.grid = grid(geo)
        # make grid here.

    def fock(self, D):
        h = self.i1.T + self.i1.V
        G = self.i2.get_2jk(D)
        XC = 0
        return h + G + XC

    def energy(self, D, F):
        h = self.h
        G = F - h # TODO: not optimal
        Eone = 2 * trace2(h, D)
        Etwo = trace2(D, G)
        Exc = 0.0 # TODO: not implemented
        return self.Enuc + Eone + Etwo + Exc


class uhf(hamiltonian):
    name = 'UHF'

    def fock(self, Da, Db):
        h = self.h
        J = self.i2.get_j(Da + Db)
        Ka, Kb = self.i2.get_k(Da), self.i2.get_k(Db)
        Ga = J - Ka
        Gb = J - Kb
        Fa = h + Ga
        Fb = h + Gb
        return Fa, Fb

    def eigenv(self, Fa, Fb):
        orbea, orbsa = geigh(Fa, self.i1.S)
        orbeb, orbsb = geigh(Fb, self.i1.S)
        return (orbea, orbsa), (orbeb, orbsb)

    def density(self, orbsa, orbsb):
        Da = dmat(orbsa, self.geo.nup())
        Db = dmat(orbsb, self.geo.ndown())
        return Da, Db

    def energy(self, Da, Db, Fa, Fb):
        h = self.h
        Ga = Fa - h # TODO: not optimal
        Gb = Fb - h # TODO: not optimal
        Eone = trace2(Da + Db, h)
        Etwo = trace2(Ga, Da)/2 + trace2(Gb, Db)/2
        return self.Enuc + Eone + Etwo


class cuhf(uhf):
    name = 'CUHF'

    def __init__(self, geo, bfs, norbsh=[], fi=[]):
        rhf.__init__(self, geo, bfs)
        self.norbsh = norbsh
        self.fi = fi


if __name__ == '__main__':
    import doctest; doctest.testmod()
