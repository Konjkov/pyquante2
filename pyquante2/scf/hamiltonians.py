from pyquante2.utils import dmat, trace2, geigh
from pyquante2.grid.grid import grid
from pyquante2.ints.integrals import onee_integrals, twoe_integrals
import numpy as np

np.set_printoptions(threshold='nan', linewidth=10000)

class hamiltonian(object):
    """ Class that represent hamiltonian manipulation methods, not state
    """
    name = 'abstract'
    def __init__(self, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        self.Enuc = bfs.atoms.nuclear_repulsion()
        self.i1 = onee_factory(bfs)
        self.i2 = twoe_factory(bfs)
        self.h = self.i1.T + self.i1.V


class rhf(hamiltonian):
    """ RHF hamiltonian
    """
    name = 'RHF'

    def __init__(self, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        hamiltonian.__init__(self, bfs, onee_factory=onee_factory, twoe_factory=twoe_factory)
        self.nocc = bfs.atoms.nocc()

    def fock(self, D):
        """ Fock matrix"""
        return self.h + self.i2.get_2jk(D)

    def energy(self, D, F):
        """ Energy"""
        return self.Enuc + trace2(self.h + F, D)

    def eigenv(self, F):
        """ Eigenvalues and Eigenvectors"""
        orbe, orbs = geigh(F, self.i1.S)
        return orbe, orbs

    def density(self, orbs):
        """ Electron density"""
        return dmat(orbs, self.nocc)


class rdft(rhf):
    """Hamiltonian for DFT calculations. Adds a grid to RHF iterator."""
    name = 'DFT'

    def __init__(self, geo, bfs):
        rhf.__init__(self, geo, bfs)
        self.grid = grid(geo)
        # make grid here.

    def fock(self, D):
        """ Fock matrix"""
        XC = 0
        return self.h + self.i2.get_2jk(D) + XC

    def energy(self, D, F):
        """ Energy"""
        Exc = 0.0 # TODO: not implemented
        return self.Enuc + trace2(self.h + F, D) + Exc


class uhf(hamiltonian):
    name = 'UHF'

    def __init__(self, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        hamiltonian.__init__(self, bfs, onee_factory=onee_factory, twoe_factory=twoe_factory)
        self.nup = bfs.atoms.nup()
        self.ndown = bfs.atoms.ndown()

    def fock(self, Da, Db):
        """ Fock matrix"""
        J = self.i2.get_j(Da + Db)
        Ka, Kb = self.i2.get_k(Da), self.i2.get_k(Db)
        Fa = self.h + J - Ka
        Fb = self.h + J - Kb
        return Fa, Fb

    def energy(self, Da, Db, Fa, Fb):
        """ Energy"""
        Ea = trace2(self.h + Fa, Da)
        Eb = trace2(self.h + Fb, Db)
        return self.Enuc + (Ea + Eb)/2

    def eigenv(self, Fa, Fb):
        orbea, orbsa = geigh(Fa, self.i1.S)
        orbeb, orbsb = geigh(Fb, self.i1.S)
        return (orbea, orbsa), (orbeb, orbsb)

    def density(self, orbsa, orbsb):
        Da = dmat(orbsa, self.nup)
        Db = dmat(orbsb, self.ndown)
        return Da, Db


class rohf(hamiltonian):
    """Reference: Takashi Tsuchimochi1 and Gustavo E. Scuseria
       J. Chem. Phys. 133, 141102 (2010)
       ROHF theory made simple. DOI 10.1063/1.3503173"""
    name = 'ROHF'

    def __init__(self, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        hamiltonian.__init__(self, bfs, onee_factory=onee_factory, twoe_factory=twoe_factory)
        self.norb = len(bfs)
        self.nup = bfs.atoms.nup()
        self.ndown = bfs.atoms.ndown()
        # Canonical form
        # self.Acc, self.Bcc, self.Aoo, self.Boo, self.Avv, self.Bvv = 0.0, 1.0, 1.0, 0.0, 1.0, 0.0
        # Guest and Saunders
        self.Acc = self.Bcc = self.Aoo = self.Boo = self.Avv = self.Bvv = 0.5
        # Roothaan single matrix  -1/2  3/2  1/2  1/2  3/2 -1/2
        # self.Acc, self.Bcc, self.Aoo, self.Boo, self.Avv, self.Bvv = -0.5, 1.5, 0.5, 0.5, 1.5, -0.5
        # Davidson                 1/2  1/2   1    0    1    0
        # self.Acc, self.Bcc, self.Aoo, self.Boo, self.Avv, self.Bvv = 0.5, 0.5, 1.0, 0.0, 1.0, 0.0
        # Binkley, Pople, Dobosh   1/2  1/2   1    0    0    1
        # self.Acc, self.Bcc, self.Aoo, self.Boo, self.Avv, self.Bvv = 0.5, 0.5, 1.0, 0.0, 0.0, 1.0
        # McWeeny and Diercksen    1/3  2/3  1/3  1/3  2/3  1/3
        # self.Acc, self.Bcc, self.Aoo, self.Boo, self.Avv, self.Bvv = 1/3.0, 2/3.0, 1/3.0, 1/3.0, 2/3.0, 1/3.0
        # Faegri and Manne         1/2  1/2   1    0   1/2  1/2
        # self.Acc, self.Bcc, self.Aoo, self.Boo, self.Avv, self.Bvv = 0.5, 0.5, 1.0, 0.0, 0.5, 0.5
        # R-matrix projection operator = D (density)

    def fock(self, Da, Db):
        """ Fock matrix"""
        J = self.i2.get_j(Da + Db)
        Ka, Kb = self.i2.get_k(Da), self.i2.get_k(Db)
        Fa = self.h + J - Ka
        Fb = self.h + J - Kb
        return Fa, Fb

    def energy(self, Da, Db, Fa, Fb):
        """ Energy"""
        Ea = trace2(self.h + Fa, Da)
        Eb = trace2(self.h + Fb, Db)
        return self.Enuc + (Ea + Eb)/2

    def effective_fock(self, Fa, Fb):
        Fc = (Fa + Fb) / 2.0
        F2 = self.Acc * Fa + self.Bcc * Fb
        F1 = self.Aoo * Fa + self.Boo * Fb
        F0 = self.Avv * Fa + self.Bvv * Fb
        # self.nup > self.ndown
        c = slice(0, self.ndown)
        o = slice(self.ndown, self.nup)
        v = slice(self.nup, self.norb)
        Feff = np.zeros((self.norb, self.norb), 'd')
        Feff[c, c], Feff[c, o], Feff[c, v] = F2[c, c], Fb[c, o], Fc[c, v]
        Feff[o, c], Feff[o, o], Feff[o, v] = Fb[o, c], F1[o, o], Fa[o, v]
        Feff[v, c], Feff[v, o], Feff[v, v] = Fc[v, c], Fa[v, o], F0[v, v]
        return Feff

    def eigenv(self, F):
        """ Eigenvalues and Eigenvectors
        """
        orbe, orbs = geigh(F, self.i1.S)
        return orbe, orbs

    def density(self, orbs):
        Da = dmat(orbs, self.nup)
        Db = dmat(orbs, self.ndown)
        return Da, Db


class cuhf(uhf):
    """Reference: Takashi Tsuchimochi1 and Gustavo E. Scuseria
       J. Chem. Phys. 133, 141102 (2010)
       ROHF theory made simple. DOI 10.1063/1.3503173"""
    name = 'CUHF'

    def __init__(self, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        uhf.__init__(self, bfs, onee_factory=onee_factory, twoe_factory=twoe_factory)
        self.norb = len(bfs)

    def fock_energy(self, Da, Db):
        # Fock
        h = self.h
        J = self.i2.get_j(Da + Db)
        Ka, Kb = self.i2.get_k(Da), self.i2.get_k(Db)
        Ga = J - Ka
        Gb = J - Kb
        Fa = h + Ga
        Fb = h + Gb
        # Energy
        Eone = trace2(Da + Db, h)
        Etwo = trace2(Ga, Da)/2 + trace2(Gb, Db)/2
        E = self.Enuc + Eone + Etwo
        # Create Effective Fock Matrixs
        P = (Da + Db)/2 # charge density matrix
        M = (Da - Db)/2 # spin density matrix
        # obtain Cno - natural orbitals

        Fc = (Fa + Fb) / 2.0
        c = slice(0, self.ndown)
        o = slice(self.ndown, self.nup)
        v = slice(self.nup, self.norb)
        Fa[c, v] = Fc[c, v]
        Fb[v, c] = Fc[v, c]
        return Fa, Fb, E


if __name__ == '__main__':
    import doctest; doctest.testmod()
