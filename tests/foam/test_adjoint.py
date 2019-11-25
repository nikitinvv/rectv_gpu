#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv_gpu
import numpy as np
import dxchange
import tomopy

def getp(a):
    return a.__array_interface__['data'][0]

def takephi(m, ntheta):
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.zeros([m, ntheta, 2], dtype='float32')
    phi[:, :, 0] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, :, 1] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[0] = 0  # symmetric
    return phi

if __name__ == "__main__":

    data = np.load("data2.npy")  # load continuous data
    rot_center = 252
    nsp = 4  # number of slices to process simultaniously by gpus
    m = 8  # number of basis functions, must be a multiple of nframes
    lambda0 = pow(2, -9)  # regularization parameter 1
    lambda1 = pow(2, 1)  # regularization parameter 2
    ngpus = 1
    [ns, ntheta, n] = data.shape
    # Make a class for tv
    cl = rectv_gpu.rectv(n, ntheta, m, ns,
                         ns, ngpus, rot_center, lambda0, lambda1)
    theta = np.linspace(0, 8*np.pi, ntheta, endpoint=False).astype('float32')
    phi = takephi(m, ntheta)
    # Run iterations
    cl.adjoint_tests(getp(data), getp(theta), getp(phi))
