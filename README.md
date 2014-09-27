# PyQuante2

## Why rewrite PyQuante?
[PyQuante](http://pyquante.sf.net) is a Quantum Chemistry Suite
written in Python. Over the years, there are many things that I'm
growing increasingly unsatisfied with:

* I learned Python (essentially) as I was writing PyQuante, which
  means that I wasn't as good a programmer when I did most of the
  writing of the core modules as I am now.
* Python was only at version 1.4-2.0 when I started writing. Many
  aspects of the language have changed substantially since then,
  especially the maturity of the numpy module.

I'm playing with PyQuante2 as a way of exploring more intelligent
design. I'm slowly moving components over from PyQuante into
PyQuante2. My goals are:

* Clean code with a consistent style reflecting Python 2.7;
* Easily understood code structure;
* Much better test suite structure, use of nosetests as a matter of
  course. 
* Extensions 
  - Written only in Cython
  - Not required for base code to function

Of course, it would be nice if this code were also much faster than
PyQuante, but I'd be surprised if this happened automatically, and I'm
prepared to deal with code performance at a later time.

There is an IPython notebook surveying some of the new features that
can be viewed as a [gist](https://gist.github.com/rpmuller/5745404) or
[on nbviewer](http://nbviewer.ipython.org/5745404). 

## Requirements:
* Python 2.7
* Numpy >= 1.7
* PyParsing 2.0.2 for basis set files parsing
* Cython 0.21
* The test suites run without matplotlib or pyglet, but there is extra
  functionality using both.

## What works so far:
* Huzinaga and HGP integral methods, both in Python and with Cython wrappers to C
* RHF, UHF wave functions
* MP2 perturbation theory
* Large limited number of basis sets available from 
  module uses basis sets downloaded from https://bse.pnl.gov/bse/portal
* Line and contour plotting
* Basic IPython notebook support for some objects

Feel free to fork this if it interests you. The PyQuante code is still
around, so I'm not rushing through the process, I'm just taking as
much time as I feel I need to do this properly.

