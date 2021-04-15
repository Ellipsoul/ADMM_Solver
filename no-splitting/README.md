# ADMM solver with no splitting
A python implementation of and ADMM solver for conic programs.
It essentially implements the approach described by [Wen et al. (2010), "Alternating direction augmented Lagrangian methods for semidefinite programming"](https://doi.org/10.1007/s12532-010-0017-1).

## Contents
* `solver.py`: the main solvers
* `helpers.py`: some useful functions and class definitions used in the main solver
* `example.py`: an example
* `pop.mat`: a data file exported from MATLAB, required for the example

## Requirements
This code solves a positive-definite sparse linear system using Cholesky factorization (with caching), as provided by [sksparse.cholmod](https://scikit-sparse.readthedocs.io/en/latest/cholmod.html). To install [sksparse](https://scikit-sparse.readthedocs.io/en/latest/overview.html#) on Ubuntu:
```
$ sudo apt-get install libsuitesparse-dev
$ pip install --user scikit-sparse
```
For other platforms, check out the [official installation instructions](https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation).
