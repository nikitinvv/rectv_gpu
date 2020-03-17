
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rectv_gpu
import numpy as np
import dxchange
import sys


def getp(a):
    return a.__array_interface__['data'][0]

def takephi(ntheta):
    m = 32 # number of basis functions
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.array(np.zeros([m, 2*ntheta], dtype='float32'),order='C')
    phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[0] = 0  # symmetric
    return phi

if __name__ == "__main__":

    data = np.array(np.load("foam.npy").astype('float32'),order='C')
    [ns, ntheta, n] = data.shape
    print([ns,ntheta,n])
    rot_center = n//2-1.5 # rotation center
    lambda0a = [1e-3,5e-3,1e-2,5e-4]  # regularization parameter 1
    lambda1a = [16]#,32,64,8]  # regularization parameter 2
    nsp = 2 # number of slices to process simultaniously by gpus
    ngpus = 4 # number of gpus
    niter = 512  # number of ADMM iterations
    titer = 4  # number of inner tomography iterations
    
    # take basis functions for decomosition 
    phi = takephi(ntheta) 
    m = phi.shape[0] # number of basis functions
    # creaate class for processing
    for k in range(len(lambda0a)):
        for j in range(len(lambda1a)):
            lambda0=lambda0a[k]
            lambda1=lambda1a[j]
            step = 4/(lambda1**2)
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
            for kk in range(rtv.shape[0]):
                dxchange.write_tiff_stack(rtv[kk], 'rec_tv'+str(lambda0)+str(lambda1)+'/rec_'+str(kk), overwrite=True)
            dxchange.write_tiff_stack(np.rot90(rtv[32],1,axes=(1,2))[:,200:350,200:450], 'rec_tvtime'+str(m)+str(lambda0)+str(lambda1)+'/rec.tiff', overwrite=True)
            #dxchange.write_tiff(np.rot90(rtv[16,7],1,axes=(1,2)), 'rec_tvtimenew/'+str(lambda0)+str(lambda1)+'.tiff', overwrite=True)
            cl=[]
