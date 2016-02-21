# COMET
Code (MATLAB) and data for the paper "Learning Sparse Metrics, One Feature at a Time", Y. Atzmon, U. Shalit, G. Chechik, JMLR 2015

[Project page](http://chechiklab.biu.ac.il/~yuvval/COMET/)

## Installation Instructions (tested on linux)
1. Download this repository or clone it using `git clone https://github.com/chechiklab/COMET`

2. Download and compile the Suitesparse (4.4.5+) Cholesky solver:

2.1 open `http://faculty.cse.tamu.edu/davis/SuiteSparse/` and download version 4.4.5 or later

2.2 unpack SuiteSparse to a directory, open MATLAB on that directory and call `SuiteSparse_install`

2.3 `cd CHOLMOD/MATLAB`, call `cholmod_demo` to test that the installation was successful

2.4 write down the full path of the location of `CHOLMOD/MATLAB` and replace it respectively under initpaths.m

## Running the examples
To run the examples with the protein dataset, one should download and install LIBSVM matlab package and add it to the matlab path.
http://www.csie.ntu.edu.tw/~cjlin/libsvm/#download

### Sparse COMET
1. Follow the instructions above to install SuiteSparse (4.4.5+) and LIBSVM. 

2. Open `example_comet_sparse_training.m` and comment-in the dataset name you would like to train upon (`'protein'` or `'RCV1_4_5K'`).

3. Execute `example_comet_sparse_training.m` from MATLAB.

### Dense COMET
1. Follow the instructions above to install SuiteSparse (4.4.5+) and LIBSVM. 

2. Open `example_comet_dense_training.m` and comment-in the dataset name you would like to train upon (`'protein'` or `'RCV1_4_5K'`).

3. Execute `example_comet_dense_training.m` from MATLAB.


## How to set the hyper-params
There are some technical clarifications need to be made about settings some of the hyper-params. I will document it during march 2016. Send me an email or open an issue on the github repository, if you would like to use this code beforehand and I will assist you.