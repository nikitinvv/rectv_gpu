#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv_gpu
import numpy as np
import dxchange
#import tomopy
import sys


def getp(a):
    return a.__array_interface__['data'][0]

def takephi(m, ntheta):
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    # phi = np.exp(2*np.pi*1j*x*y/ntheta)/np.sqrt(ntheta)
    phi = np.zeros([m, 2*ntheta], dtype='float32')
    phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[0] = 0  # symmetric
    return phi

if __name__ == "__main__":

    data = np.load("data.npy")  # load continuous data
    [ns, ntheta, n] = data.shape
    rot_center = n/2
   
    m = 8  # number of basis functions, must be a multiple of nframes
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
    phi = takephi(m, ntheta)
    # memory for result
    rtv = np.zeros([ns,m,n,n], dtype='float32')
    # Run iterations
    cl.run(getp(rtv), getp(data), getp(theta), getp(phi),  niter, titer)
    print(np.linalg.norm(rtv))
    for k in range(rtv.shape[0]):
        dxchange.write_tiff_stack(rtv[k], 'rec_tv/rec_'+str(k), overwrite=True)
