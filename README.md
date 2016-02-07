# COMET
Sources (MATLAB) for the paper Atzmon, Shalit and Chechik, "Learning Sparse Metrics, One Feature at a Time", JMLR 2015

## Installation Instructions (tested on linux)
1. Download this repository or clone it using `git clone https://github.com/chechiklab/COMET`
2. Download and compile the Suitesparse (4.4.5+) Cholesky solver:
2.1 open `http://faculty.cse.tamu.edu/davis/SuiteSparse/` and download version 4.4.5 or later
2.2 unpack SuiteSparse to a directory, open MATLAB on that directory and call `SuiteSparse_install`
2.3 `cd CHOLMOD/MATLAB`, call `cholmod_demo` to test that the installation was successful
2.4 write down the full path of the location of `CHOLMOD/MATLAB` and replace it respectively under initpaths.m



