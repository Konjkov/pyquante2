#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True


classifiers = """\
Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
"""
    
cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("pyquante2.cutils",["cython/cutils.pyx"]),
        Extension("pyquante2.cone",["cython/cone.pyx","cython/cints.c"]),
        Extension("pyquante2.ctwo",["cython/ctwo.pyx","cython/cints.c","cython/chgp.c"]),
        Extension("pyquante2.clibint", ["cython/clibint.pyx"],
                  include_dirs=['/usr/include/libint', '/usr/include/libderiv', '/usr/include/libr12', np.get_include()],
                  libraries=['int', 'deriv', 'r12'],
                  library_dirs=['/usr/lib'],
              ),
        Extension("pyquante2.cbecke",["cython/cbecke.pyx"],
                   include_dirs=[np.get_include()])
        ]
else:
    ext_modules += [
        Extension("pyquante2.cutils",["cython/cutils.c"]),
        Extension("pyquante2.cone",["cython/cone.c","cython/cints.c"]),
        Extension("pyquante2.ctwo",["cython/ctwo.c","cython/cints.c","cython/chgp.c"]),
        ]

with open('README.md') as file:
    long_description = file.read()

setup(name='pyquante2',
      version='0.1',
      description='Python Quantum Chemistry, version 2.0',
      long_description = long_description,
      author='Rick Muller',
      author_email='rpmuller@gmail.com',
      url='http://pyquante.sourceforge.net',
      install_requires=['numpy'],
      packages=['pyquante2',
                'pyquante2.basis',
                'pyquante2.dft',
                'pyquante2.geo',
                'pyquante2.graphics',
                'pyquante2.grid',
                'pyquante2.ints',
                'pyquante2.pt',
                'pyquante2.cc',
                'pyquante2.scf',
                'pyquante2.viewer',
                ],
      package_data={'pyquante2.basis': ['libraries/*']},
      ext_modules = cythonize(ext_modules),
      )
