#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv_gpu
import numpy as np
import dxchange
import sys
import tomocg as pt

def getp(a):
    return a.__array_interface__['data'][0]

def takephi(ntheta):
    m = 8  # number of basis functions
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.zeros([m, 2*ntheta], dtype='float32')
    phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[0] = 0  # symmetric
    return phi

if __name__ == "__main__":

    data = np.array(np.load("prjbin1.npy").astype('float32').swapaxes(0,1),order='C')
    [ns, ntheta, n] = data.shape
    rot_center = (1168-456)//2
    lambda0 = 1e-12  # regularization parameter 1
    lambda1 = 4  # regularization parameter 2
    nsp = 4 # number of slices to process simultaniously by gpus
    ngpus = 1 # number of gpus
    niter = 8  # number of ADMM iterations
    titer = 4  # number of inner tomography iterations
    
    # take basis functions for decomosition 
    phi = takephi(ntheta) 
    m = phi.shape[0] # number of basis functions
    # creaate class for processing
    cl = rectv_gpu.rectv(n, ntheta, m, ns,
                         nsp, ngpus, rot_center, lambda0, lambda1)
    # angles
    theta = np.array(np.load('theta.npy')[:data.shape[1]].astype('float32'),order='C')
    u = np.array(np.zeros([ns,n,n], dtype='complex64'), order='C')
    #for k in range(8):
        #with pt.SolverTomo(theta[k*200:(k+1)*200], ntheta//8, ns, n, nsp, rot_center) as slv:
            #print(k)
            ### generate data
            ##u = np.zeros([ns,n,n],dtype='complex64')
            #u = slv.cg_tomo_batch(data.swapaxes(0,1)[k*200:(k+1)*200]+1j*0,u,4)
            #### memory for result\
    #exit()
    rtv = np.array(np.zeros([ns,m,n,n], dtype='float32'), order='C')
    # Run iterations
    dbg = True # show relative convergence
    cl.run(getp(rtv), getp(data), getp(theta), getp(phi),  niter, titer, dbg)
    # Save result
    for k in range(rtv.shape[0]):
         dxchange.write_tiff_stack(rtv[k], 'rec_tv/rec_'+str(k), overwrite=True)
