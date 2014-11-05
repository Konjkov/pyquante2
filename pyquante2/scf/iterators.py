from guess import core_hamiltonian_rhf, core_hamiltonian_uhf, core_hamiltonian_rohf
import numpy as np


class SCFIterator(object):
    def __init__(self, H, guess=core_hamiltonian_rhf):
        self.H = H
        self.D = guess(H)
        self.converged = False

    def converge(self, tol=1e-7, maxiters=100):
        E = 0
        D = self.D
        for iteration in range(maxiters):
            """Update orbital energies and eigenvectors"""
            (F, E), Eold = self.H.fock_energy(D), E
            orbe, orbs = self.H.eigenv(F)
            D = self.H.density(orbs)
            if abs(E-Eold) < tol:
                self.D = D
                self.orbe, self.orbs = orbe, orbs
                self.energy = E
                self.converged = True
                break


class USCFIterator(object):
    def __init__(self, H, guess=core_hamiltonian_uhf):
        self.H = H
        self.Da, self.Db = guess(H)
        self.converged = False

    def converge(self, tol=1e-7, maxiters=100):
        E = 0
        Da = self.Da
        Db = self.Db
        for iteration in range(maxiters):
            (Fa, Fb, E), Eold = self.H.fock_energy(Da, Db), E
            (orbea, orbsa), (orbeb, orbsb) = self.H.eigenv(Fa, Fb)
            Da, Db = self.H.density(orbsa, orbsb)
            if abs(E-Eold) < tol:
                self.Da, self.Db = Da, Db
                self.orbea, self.orbsa = (orbea, orbsa)
                self.orbeb, self.orbsb = (orbeb, orbsb)
                self.energy = E
                self.converged = True
                break


class ROSCFIterator(object):
    def __init__(self, H, guess=core_hamiltonian_rohf):
        self.H = H
        self.Da, self.Db = guess(H)
        self.converged = False

    def converge(self, tol=1e-7, maxiters=100):
        E = 0
        Da = self.Da
        Db = self.Db
        for iteration in range(maxiters):
            (Fa, Fb, E), Eold = self.H.fock_energy(Da, Db), E
            Feff = self.H.effective_fock(Fa, Fb)
            orbe, orbs = self.H.eigenv(Feff)
            Da, Db = self.H.density(orbs)
            if abs(E-Eold) < tol:
                self.Da, self.Db = Da, Db
                self.orbe, self.orbs = (orbe, orbs)
                self.energy = E
                self.converged = True
                break


class AveragingIterator(SCFIterator):
    def converge(self, fraction=0.5, tol=1e-7, maxiters=100):
        E = 0
        D = Dold = self.D
        for iteration in range(maxiters):
            (F, E), Eold = self.H.fock_energy(D), E
            orbe, orbs = self.H.eigenv(F)
            D = self.H.density(orbs)
            D, Dold = (1-fraction)*Dold + fraction*D, D
            if abs(E-Eold) < tol:
                self.D = D
                self.orbe, self.orbs = orbe, orbs
                self.energy = E
                self.converged = True
                break


if __name__ == '__main__':
    import doctest; doctest.testmod()
