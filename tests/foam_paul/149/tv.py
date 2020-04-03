
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

    data = np.array(np.load("foam100.npy")[:,:,8:-8].astype('float32'),order='C')
    [ns, ntheta, n] = data.shape
    print([ns,ntheta,n])
    rot_center = n//2-1.5 # rotation center
    lambda0a = [1e-3]  # regularization parameter 1
    lambda1a = []  # regularization parameter 2
    ma = [16,32,8]
    nsp = 1 # number of slices to process simultaniously by gpus
    ngpus = 4 # number of gpus
    niter = 2048  # number of ADMM iterations
    titer = 4  # number of inner tomography iterations
        
    # take basis functions for decomosition 
   
  #  m = phi.shape[0] # number of basis functions
    # creaate class for processing
    for i in range(len(ma)):
        for k in range(len(lambda0a)):
            # for j in range(len(lambda1a)):
                m = ma[i]
                lambda0=lambda0a[k]
                lambda1=m#lambda1a[j]
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
                #rtv = rtv[:,:,rtv.shape[2]//6:-rtv.shape[2]//6,rtv.shape[2]//6:-rtv.shape[2]//6]
                for kk in range(rtv.shape[1]):
                    dxchange.write_tiff_stack(rtv[:,kk], 'resrec_tv'+str(m)+str(lambda0)+str(lambda1)+'/rec_'+str(kk)+'_', overwrite=True)
                # dxchange.write_tiff_stack(np.rot90(rtv[32],1,axes=(1,2))[:,200:350,200:450], 'rec_tvtime'+str(m)+str(lambda0)+str(lambda1)+'/rec.tiff', overwrite=True)
                #dxchange.write_tiff(np.rot90(rtv[16,7],1,axes=(1,2)), 'rec_tvtimenew/'+str(lambda0)+str(lambda1)+'.tiff', overwrite=True)
                cl=[]
