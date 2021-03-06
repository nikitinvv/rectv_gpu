# rectv_gpu
# Four-dimensional tomographic reconstruction by time domain decomposition

## Clone git repository

`git clone https://github.com/nikitinvv/rectv_gpu`

## Installation MAX IV

`module add  GCC/8.2.0-2.31.1 icc/2019.1.144-GCC-8.2.0-2.31.1 ifort/2019.1.144-GCC-8.2.0-2.31.1 CUDA/10.1.105`

`conda install -c conda-forge dxchange scikit-build matplotlib notebook ipywidgets`

`cd rectv_gpu; python setup.py install`

## Use MAX IV gn1-3 nodes for remote jupyter notebook

Allocate GPU resources:

`salloc -p v100`

In remote host (e.g. gn1), open the terminal, change directory to where you have your notebooks and type:

`jupyter notebook --no-browser --port=<port1>`

E.g. port1=13543

In your local computer type:

`ssh -N -f -L localhost:<port2>:localhost:<port1> username@address`

E.g. port2=13545, address=w-picard05-clu0-gn-1.maxiv.lu.se

Now open web browser (google chrome, firefox, ...) and type:

`localhost:<port2>`

## Examples with jupyter notebook

See examples/
