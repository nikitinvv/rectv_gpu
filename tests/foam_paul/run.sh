#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -J tomo4d
#SBATCH -p v100
#SBATCH --exclude gn1
#SBATCH -c 40
#SBATCH --mem 160G
#SBATCH -o p_rec_gpu.out
#SBATCH -e p_rec_gpu.err


module add GCC/8.3.0  GCCcore/8.3.0  CUDA/10.1.243 OpenMPI/3.1.4

python tv.py
