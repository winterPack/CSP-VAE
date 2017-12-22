# CSP-VAE
A 3D Variational AutoEncoder for MD simulation data

## Requirements:
  Python 2
  
  [Pytorch](http://pytorch.org/)
  
## How-to's:

In order to run on NCSU hpc, follow the steps below.

1. Download [virtualenv](https://github.com/pypa/virtualenv). Follow instructions to create a virtualenv. (Otherwise, NCSU does not allow users to install python packages.)
2. Activate virtualenv and install Pytorch, see [here](http://pytorch.org/).
3. git clone files to working directory.
4. use hpc.sub to submit jobs.

To change neural network structures, check model.py.

To change hyperparameters, check main.py.
