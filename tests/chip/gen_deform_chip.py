import dxchange
import numpy as np
import tomocg as tc
import elasticdeform
import concurrent.futures as cf
from itertools import repeat
from functools import partial

import threading
import matplotlib.pyplot as plt
import os 
from scipy import ndimage
def fwd_deform_tomo(u0,theta,disp0, dstart,id):
      print(id)
      n = u0.shape[2]
      nz = u0.shape[0]
      displacement = disp0*(id-dstart)/128*5
      u = elasticdeform.deform_grid(
            u0, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
      dxchange.write_tiff(u[25],'u/u'+str(id)+'.tiff', overwrite=True)                  
      u = u+1j*0
      with tc.SolverTomo(theta[id:id+1], 1, nz, n, 128, 64) as tslv:
            psi = tslv.fwd_tomo_batch(u).real
      return psi
if __name__ == "__main__":

      # Model parameters
      n = 128  # object size n x,y
      nz = 128  # object size in z
      ntheta = 8*128  # number of angles (rotations)
      theta = np.linspace(0, 8*np.pi, ntheta,endpoint=False).astype('float32')  # angles
      # Load object
      u0 = dxchange.read_tiff(
            'data/delta-chip-128.tiff')#[:, 64:64+ntheta].swapaxes(0, 1)                  
      points = [3, 3, 3]
      disp0 = (np.random.rand(3, *points) - 0.5)
      psi = np.zeros([ntheta,nz,n],dtype='float32')
      
      dstart = 128*4
      dend = 128*6
      # part 1
      u = u0+1j*0
      with tc.SolverTomo(theta[0:dstart], dstart, nz, n, 128, 64) as tslv:
            psi[:dstart] = tslv.fwd_tomo_batch(u).real   
      # part 3            
      displacement = disp0*(dend-dstart)/128*5
      u = elasticdeform.deform_grid(
            u0, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
      u = u+1j*0
      with tc.SolverTomo(theta[dend:], ntheta-dend, nz, n, 128, 64) as tslv:
            psi[dend:] = tslv.fwd_tomo_batch(u).real
      # part 2
      
      with cf.ThreadPoolExecutor(32) as e:
            shift = dstart
            for psi0 in e.map(partial(fwd_deform_tomo, u0, theta,disp0,dstart), range(dstart, dend)):
                psi[shift] = psi0
                shift += 1            
      dxchange.write_tiff_stack(psi,'psi/psi.tiff',overwrite=True)                  
      np.save('datachip',psi.swapaxes(0,1))
