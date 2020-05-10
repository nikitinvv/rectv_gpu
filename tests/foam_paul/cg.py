#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tomocg as pt
import numpy as np
import dxchange
import sys

if __name__ == "__main__":

    #data = np.load("foam.npy").swapaxes(0,1).astype('complex64')[:,:,8:-8]
    [ntheta, ns, n] = [192*8,128,512]
    center = n//2 # rotation center
    nsp = ns # number of slices to process simultaniously by gpus
    ngpus = 4 # number of gpus
    
    theta = np.array(np.linspace(0,8*np.pi,192*8,endpoint=False).astype('float32'),order='C')
    print((theta[2]-theta[1])*180/np.pi)
    ntheta =  len(theta)
    alphaa = [50,100,150]
    for kk in range(len(alphaa)):
        data = np.array(np.load('foam'+str(alphaa[kk])+'.npy')[:,:,8:-8].swapaxes(0,1).astype('complex64'),order='C')
        for k in range(8):
            with pt.SolverTomo(theta[k*192:(k+1)*192], 192, ns, n, ns, center) as slv:
                print(k)
                # generate data
                datae = data[k*192:(k+1)*192]
                print(datae.shape)
                # initial guess
                u = np.zeros([ns,n,n],dtype='complex64')                    
                dxchange.write_tiff(datae.real,  'datagen.tiff', overwrite=True)
                u = slv.cg_tomo_batch(datae,u,64)
                #dxchange.write_tiff_stack(u.real,  'rec'+str(k)+'/u.tiff', overwrite=True)
                #dxchange.write_tiff(u[16].real,  'rectime/u'+str(k)+'.tiff', overwrite=True)
                #u = u[:,u.shape[2]//2-528//2:u.shape[2]//2+528//2,u.shape[2]//2-528//2:u.shape[2]//2+528//2]
                    
                dxchange.write_tiff_stack(np.rot90(u.real,-1,axes=(1,2)), 'reccga'+str(alphaa[kk])+'/rec_'+str(k), overwrite=True)
                #dxchange.write_tiff(u.real[32,:,:], 'reccg/rec'+str(alphaa[kk])+'.tiff', overwrite=True)