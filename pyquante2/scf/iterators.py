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
            F = self.H.fock(D)
            E, Eold = self.H.energy(D, F), E
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
            Fa, Fb = self.H.fock(Da, Db)
            E, Eold = self.H.energy(Da, Db, Fa, Fb), E
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
        orbs = None
        for iteration in range(maxiters):
            Fa, Fb = self.H.fock(Da, Db)
            E, Eold = self.H.energy(Da, Db, Fa, Fb), E
            Feff = self.H.effective_fock(Fa, Fb, orbs)
            orbe, orbs = self.H.eigenv(Feff)
            Da, Db = self.H.density(orbs)
            if abs(E-Eold) < tol:
                self.Da, self.Db = Da, Db
                self.orbe, self.orbs = (orbe, orbs)
                self.energy = E
                self.converged = True
                break


class ROHFIterator(SCFIterator):
    def __init__(self,H,c=None,tol=1e-5,maxiters=100):
        SCFIterator.__init__(self,H,c,tol,maxiters)
        self.nup,self.ndown = self.H.geo.nup(),self.H.geo.ndown()
        return

    def __next__(self):
        self.iterations += 1
        if self.iterations > self.maxiters:
            raise StopIteration
        Dup = dmat(self.c,self.nup)
        Ddown = dmat(self.c,self.ndown)
        self.c = self.H.update(Dup,Ddown,self.c)
        E = self.H.energy
        if abs(E-self.Eold) < self.tol:
            self.converged = True
            raise StopIteration
        self.Eold = E
        return E

class AveragingIterator(SCFIterator):
    def converge(self, fraction=0.5, tol=1e-7, maxiters=100):
        E = 0
        D = Dold = self.D
        for iteration in range(maxiters):
            F = self.H.fock(D)
            E, Eold = self.H.energy(D, F), E
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
