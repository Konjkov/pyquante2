from pyquante2.grid.grid import grid
from pyquante2.ints.integrals import onee_integrals, twoe_integrals
from pyquante2.utils import trace2, geigh
from pyquante2.scf.iterators import SCFIterator, USCFIterator, AveragingIterator
import numpy as np

class hamiltonian(object):
    name = 'abstract'
    def __init__(self, geo, bfs, onee_factory=onee_integrals, twoe_factory=twoe_integrals):
        self.geo = geo
        self.bfs = bfs
        self.Enuc = self.geo.nuclear_repulsion()
        self.i1 = onee_factory(bfs,geo)
        self.i2 = twoe_factory(bfs)
        self.energies = []
        self.energy = 0
        self.converged = False

    def __repr__(self):
        lines = ["%s Hamiltonian" % self.name]
        lines.append(str(self.geo))
        lines.append("Basis set: %s, Nbf: %d" %  (self.bfs.name,len(self.bfs)))
        lines.append("Status: Converged = %s" % self.converged)
        for i,E in enumerate(self.energies):
            lines.append("%d  %.5f" % (i,E))
        return "\n".join(lines)

    def _repr_html_(self):
        import xml.etree.ElementTree as ET
        top = ET.Element("html")
        h2 = ET.SubElement(top,"h2")
        h2.text = "%s Hamiltonian" % self.name
        top.append(self.geo.html())
        p = ET.SubElement(top,"p")
        p.text = "Basis set: %s, Nbf: %d" % (self.bfs.name,len(self.bfs))
        p = ET.SubElement(top,"p")
        p.text = "Status: Converged=%s" % self.converged
        if self.energies:
            table = ET.SubElement(top,"table")
            tr = ET.SubElement(table,"tr")
            for heading in ["#","Energy"]:
                td = ET.SubElement(tr,"th")
                td.text = heading
            for i,energy in enumerate(self.energies):
                tr = ET.SubElement(table,"tr")
                td = ET.SubElement(table,"td")
                td.text = str(i)
                td = ET.SubElement(table,"td")
                td.text = "%.5f" % energy
        return ET.tostring(top)

    def converge(self,iterator=SCFIterator,**kwargs):
        converger = iterator(self,**kwargs)
        self.energies = []
        for en in converger:
            self.energies.append(en)
        self.converged = converger.converged
        return self.energies

    def update(self,*args,**kwargs): raise Exception("Unimplemented")

class rhf(hamiltonian):
    """
    >>> from pyquante2.geo.samples import h2
    >>> from pyquante2.basis.basisset import basisset
    >>> bfs = basisset(h2,'sto-3g')
    >>> h2_rhf = rhf(h2,bfs)
    >>> ens = h2_rhf.converge(SCFIterator)
    >>> np.isclose(h2_rhf.energy,-1.11709942949)
    True

    >>> ens = h2_rhf.converge(AveragingIterator,maxiters=100)
    >>> np.isclose(h2_rhf.energy,-1.11709325545)
    True

    >>> ens = h2_rhf.converge(SCFIterator,maxiters=1)
    >>> np.isclose(h2_rhf.energy,0.485554)
    True
    >>> h2_rhf.converged
    False
    """
    name = 'RHF'

    def update(self, D):
        h = self.i1.T + self.i1.V    # 1-e Hamiltonian
        G = self.i2.get_2jk(D)       # 2-e part of Fock matrix
        F = h + G                    # Fock matrix
        self.orbe, self.orbs = geigh(F, self.i1.S)
        Eone = 2 * trace2(h, D)
        Etwo = trace2(D, G)
        self.energy = self.Enuc + Eone + Etwo
        return self.orbs

class rdft(rhf):
    "Hamiltonian for DFT calculations. Adds a grid to RHF iterator."
    name = 'DFT'

    def __init__(self, geo, bfs):
        rhf.__init__(self,geo,bfs)
        self.grid = grid(geo)
        # make grid here.

    def update(self,D):
        h = self.i1.T + self.i1.V
        J = self.i2.get_2j(D)
        F = H + J

        # XC = ???
        # F = F + XC

        self.orbe, self.orbs = geigh(F, self.i1.S)
        Eone = 2 * trace2(h, D)
        Etwo = trace2(D, G)
        Exc = 0.0
        self.energy = self.Enuc + Eone + Etwo + Exc
        return self.orbs

class rohf(rhf):
    """Hamiltonian for ROHF calculations. Adds shells information
    >>> from pyquante2.geo.samples import h2
    >>> from pyquante2.basis.basisset import basisset
    >>> bfs = basisset(h2,'sto-3g')
    >>> h2_singlet = rohf(h2,bfs,[1],[1])
    >>> h2_triplet = rohf(h2,bfs,[1,1],[0.5,0.5])
    """
    name = 'ROHF'

    def __init__(self,geo,bfs,norbsh=[],fi=[]):
        rhf.__init__(self,geo,bfs)
        self.norbsh = norbsh
        self.fi = fi


class uhf(hamiltonian):
    """
    >>> from pyquante2.geo.samples import oh
    >>> from pyquante2.basis.basisset import basisset
    >>> from pyquante2.scf.iterators import USCFIterator
    >>> bfs = basisset(oh,'sto-3g')
    >>> solver = uhf(oh,bfs)
    >>> ens = solver.converge(USCFIterator)
    >>> np.isclose(solver.energy,-74.146669)
    True
    """
    name = 'UHF'

    def converge(self,iterator=USCFIterator,**kwargs):
        return hamiltonian.converge(self,iterator,**kwargs)

    def update(self, Da, Db):
        h = self.i1.T + self.i1.V
        J = self.i2.get_j(Da+Db)
        Ka, Kb = self.i2.get_k(Da), self.i2.get_k(Db)
        Ga = J - Ka
        Gb = J - Kb
        Fa = h + Ga
        Fb = h + Gb
        self.orbea, self.orbsa = geigh(Fa, self.i1.S)
        self.orbeb, self.orbsb = geigh(Fb, self.i1.S)
        Eone = trace2(Da+Db, h)
        Etwo = trace2(Ga, Da)/2 + trace2(Gb, Db)/2
        self.energy = self.Enuc + Eone + Etwo
        return self.orbsa, self.orbsb

if __name__ == '__main__':
    import doctest; doctest.testmod()
