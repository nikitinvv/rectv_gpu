
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv_gpu
import numpy as np
import dxchange
import sys


def getp(a):
    return a.__array_interface__['data'][0]

def takephi(ntheta,m):
 #   m = 32 # number of basis functions
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.array(np.zeros([m, 2*ntheta], dtype='float32'),order='C')
    phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[0] = 0  # symmetric
    return phi

if __name__ == "__main__":

    data = np.array(np.load("foam100.npy")[:,:,14:-14].astype('float32'),order='C')
    [ns, ntheta, n] = data.shape
    print([ns,ntheta,n])
    rot_center = n//2 # rotation center
    lambda0a = [1e-3]  # regularization parameter 1
    lambda1a = [16]  # regularization parameter 2
    ma = [32]
    nsp = 1 # number of slices to process simultaniously by gpus
    ngpus = 4 # number of gpus
    niter = 2048  # number of ADMM iterations
    titer = 4  # number of inner tomography iterations
        
    # take basis functions for decomosition 
    for i in range(len(ma)):
        for k in range(len(lambda0a)):
            for j in range(len(lambda1a)):
                m = ma[i]
                lambda0=lambda0a[k]
                lambda1=lambda1a[j]
                step = 0.5
                phi = takephi(ntheta,m) 
                cl = rectv_gpu.rectv(n, ntheta, m, ns,
                            nsp, ngpus, rot_center, lambda0, lambda1, step)
                # angles
                theta = np.array(np.linspace(0, 8*np.pi, ntheta, endpoint=False).astype('float32'),order='C')
                # memory for result
                rtv = np.array(np.zeros([ns,m,n,n], dtype='float32'),order='C')
                # Run iterations
                dbg = True # show relative convergence
                cl.run(getp(rtv), getp(data), getp(theta), getp(phi),  niter, titer, dbg)
                # Save result
                for kk in range(rtv.shape[1]):
                    dxchange.write_tiff_stack(rtv[:,kk], 'resrec_tv'+str(m)+str(lambda0)+str(lambda1)+'/rec_'+str(kk)+'_', overwrite=True)
                cl=[]
