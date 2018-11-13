# rectv_gpu
# Four-dimensional tomographic reconstruction by time domain decomposition

# Installation

## Building python modules
Set `CUDAHOME` environmental variable, run

`python setup.py install`

## Simple reconstruction scenario
Read, filter, normalize data and save it to file 'data.npy'

`python read_continuous`

Reconstruct with the time-domain decompositon + regularization
`python rec_simple.py`


## Use as a module 
See an example in tomobank https://tomobank.readthedocs.io/en/latest/source/data/docs.data.dynamic.html#foam-data

`python tomopy_rectv.py dk_MCFG_1_p_s1_.h5 --type subset --nsino 0.75 --binning 2 --tv True --frame 95`

`--type` - reconstruction type (slice,subset,full)

`--nsino` - location of the sinogram used by slice or subset reconstruction (0 top, 1 bottom)

`--binning` - factor for data downsampling (0,1,2)

`--tv` - use tv reconstruction (True,False)

`--frame` - central time frame for reconstruction, 8 time frames will be reconstructed by default. Example `--frame 95` gives time frames [91,99)
