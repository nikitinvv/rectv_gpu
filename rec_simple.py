#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv
import numpy as np
import dxchange

def rec(data,m,nsp,
           lambda0,lambda1,niters,ngpus):
    """
    Reconstruct. Time-domain decomposition + regularization.
    """

    [nframes, nproj, ns, n] = data.shape

    # reorder input data for compatibility
    data = np.reshape(data,[nframes*nproj,ns,n])
    data = np.ndarray.flatten(data.swapaxes(0, 1))
    rec = np.zeros([n*n*ns*m], dtype='float32')  # memory for result

    # Make a class for tv
    cl = rectv.rectv(n, nframes*nproj, m, nframes, ns,
                     ns, ngpus, lambda0, lambda1)
    # Run iterations
    cl.itertvR_wrap(rec, data, niters)

    rec = np.rot90(np.reshape(rec, [ns, m, n, n]).swapaxes(0, 1), axes=(
        2, 3))/nproj*2  # reorder result for compatibility with tomopy
    
    # take slices corresponding to angles k\pi
    rec = rec[::m//nframes]
    
    return rec


if __name__ == "__main__":
   
    data = np.load("data.npy") # load continuous data
   
    nsp = 4  # number of slices to process simultaniously by gpus
    m = 8  # number of basis functions, must be a multiple of nframes
    lambda0 = pow(2, -9)  # regularization parameter 1
    lambda1 = pow(2, 2)  # regularization parameter 2
    niters = 1024  # number of iterations
    ngpus = 1  # number of gpus

    rec = rec(data,m,nsp,lambda0,lambda1,niters,ngpus)

    for k in range(rec.shape[0]):
        dxchange.write_tiff_stack(rec[k],'rec/rec_'+str(k))
