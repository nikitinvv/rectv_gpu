# rectv_gpu
# Four-dimensional tomographic reconstruction by time domain decomposition

# Installation
make - for building an executable file

python setup.py install  - for building  python modules

# Execution
./rectv pars64 gbubbles64 rec64

python tomopy_rec.py /home/beams0/VNIKITIN/tomobank_rec/dk_MCFG_1_p_s1_.h5 --type full --binning 2 --algorithm_type tv --frame 92 

