#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv_gpu
import numpy as np
import dxchange
import tomopy
import sys


def takephi(m, ntheta):
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.exp(2*np.pi*1j*x*y/ntheta)/np.sqrt(ntheta)
    phi = np.zeros([m, ntheta, 2], dtype='float32')
    phi[:, :, 0] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, :, 1] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[0] = 0  # symmetric
    return phi

if __name__ == "__main__":

    data = np.load("data2.npy")  # load continuous data
    # np.save('data2.npy',np.reshape(data,[300*8,4,504]).swapaxes(0,1))
    # exit()
    [ns, ntheta, n] = data.shape
    rot_center = n/2
   
    m = 16  # number of basis functions, must be a multiple of nframes
    lambda0 = 1e-3  # regularization parameter 1
    lambda1 = 1  # regularization parameter 2
    nsp = np.int(sys.argv[1]) # number of slices to process simultaniously by gpus
    ngpus = np.int(sys.argv[2]) # number of gpus
    niter = np.int(sys.argv[3])  # number of iterations
    titer = np.int(sys.argv[4])  # number of iterations
    
    cl = rectv_gpu.rectv(n, ntheta, m, ns,
                         nsp, ngpus, rot_center, lambda0, lambda1)
    # angles
    theta = np.linspace(0, 8*np.pi, ntheta, endpoint=False).astype('float32')
    # basis
    phi = takephi(m, ntheta).flatten()
    # memory for result
    rtv = np.zeros([n*n*ns*m], dtype='float32')
    data = data.flatten()
    # Run iterations
    cl.run_wrap(rtv, data, theta, phi,  niter, titer)
    rtv = np.reshape(rtv, [ns, m, n, n])
    print(np.linalg.norm(rtv))
    for k in range(rtv.shape[0]):
        dxchange.write_tiff_stack(rtv[k], 'rec_tv/rec_'+str(k), overwrite=True)
