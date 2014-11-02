"""\
 cgbf.py Perform basic operations over contracted gaussian basis
  functions. Uses the functions in pgbf.py.

 References:
  OHT = K. O-ohata, H. Taketa, S. Huzinaga. J. Phys. Soc. Jap. 21, 2306 (1966).
  THO = Taketa, Huzinaga, O-ohata, J. Phys. Soc. Jap. 21,2313 (1966).

 This program is part of the PyQuante quantum chemistry program suite
"""

import numpy as np
from pyquante2.utils import fact2

class cgbf(object):
    """
    Class for a contracted Gaussian basis function
    >>> s = cgbf(exps=[1],coefs=[1])
    >>> print(s)
    cgbf((0.0, 0.0, 0.0),(0, 0, 0),[1.0],[1.0000000000000002])
    >>> np.isclose(s(0,0,0),0.712705)
    True
    """
    contracted = True
    def __init__(self, origin=(0,0,0), powers=(0,0,0), exps=[], coefs=[]):
        assert len(origin)==3
        assert len(powers)==3

        self.origin = np.asarray(origin, 'd')
        self.powers = powers
        # TODO: temporary
        self.is_first = (powers[1] + powers[2] == 0)
        self.nfunc = (sum(powers)+1)*(sum(powers)+2)/2

        self.coefs = np.asarray(coefs, 'd')
        self.pexps = np.asarray(exps, 'd')

        self.pgbfs = []
        for expn, coef in zip(exps, coefs):
            self.add_pgbf(expn, coef, False)

        self.pnorms = np.array([self.normalization(expn) for expn in self.pexps], 'd')

    def normalization(self, expn):
        l,m,n = self.powers
        result = pow(2, 2*(l+m+n)+1.5) * pow(expn, l+m+n+1.5) / pow(np.pi, 1.5)
        result /= fact2(2*l-1) * fact2(2*m-1) * fact2(2*n-1)
        return np.sqrt(result)

    def __getitem__(self, item):
        return list(zip(self.coefs, self.pgbfs)).__getitem__(item)

    def __call__(self, *args, **kwargs):
        return sum(c*p(*args,**kwargs) for c,p in self)

    def __repr__(self):
        return "cgbf(%s,%s,%s,%s)" % (tuple(self.origin), self.powers, list(self.pexps), list(self.coefs))

    def mesh(self,xyzs):
        """
        Evaluate basis function on a mesh of points *xyz*.
        """
        return sum(c*p.mesh(xyzs) for c,p in self)

    def add_pgbf(self, expn, coef, renormalize=True):
        from pyquante2.basis.pgbf import pgbf

        self.pgbfs.append(pgbf(expn, self.origin,self.powers))

        if renormalize:
            self.normalize()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
