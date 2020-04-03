#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tomocg as pt
import numpy as np
import dxchange
import sys

if __name__ == "__main__":

    data = np.load('foam100.npy').swapaxes(0,1)[:,64:65]
    print(data.shape)
    [ntheta, ns, n] = [192*8,1,528]
    center = n//2 # rotation center
    nsp = ns # number of slices to process simultaniously by gpus
    ngpus = 4 # number of gpus
    
    theta = np.array(np.linspace(0,8*np.pi,192*8,endpoint=False).astype('float32'),order='C')
    print((theta[2]-theta[1])*180/np.pi)
    ntheta =  len(theta)
    
    for k in range(1):
        for j in np.arange(-5,5,0.5):
            with pt.SolverTomo(theta[k*192:(k+1)*192], 192, ns, n, ns, center+j) as slv:
                print(j)
                # generate data
                datae = data[k*192:(k+1)*192]
                # initial guess
                u = np.zeros([ns,n,n],dtype='complex64')                    
                u = slv.cg_tomo_batch(datae,u,32)
                dxchange.write_tiff(u.real, 'test_center/rec'+str(center+j)+'.tiff', overwrite=True)