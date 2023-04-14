#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rectv_gpu
import numpy as np
import h5py
import dxchange
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
#get_ipython().run_line_magic('matplotlib', 'inline')


# Initiate basis functions for function decomposition $u=\sum_{j=0}^{m-1}u_j\varphi_j$

# In[2]:


def takephi(ntheta):
    m = 16  # number of basis functions
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.zeros([m, 2*ntheta], dtype='float32')
    phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    return phi


# In[3]:


# with h5py.File('/data/2023-04/Sun/dark_fields_Ren_ink_1perc_0p008_23_030.h5','r') as fid:
#     dark = fid['/exchange/data_dark'][:,:32]

# with h5py.File('/data/2023-04/Sun/flat_fields_Ren_ink_1perc_0p008_23_030.h5','r') as fid:
#     flat = fid['/exchange/data_white'][:,:32]    

# with h5py.File('/data/2023-04/Sun/Ren_ink_1perc_0p008_23_030.h5','r') as fid:
#     data = fid['/exchange/data'][:90*16,:32]        
#     theta = fid['/exchange/theta'][:90*16]*0.9977/180*np.pi
# data = -np.log((data-np.mean(dark,axis=0))/(1e-3+np.mean(flat,axis=0)-np.mean(dark,axis=0))).swapaxes(0,1)


with h5py.File('/data/2023-04/Sun_rec/Ren_ink_1perc_0p008_23_030_rec.h5','r') as fid:
    data = fid['/exchange/data'][:90*16,:128]        
    theta = fid['/exchange/theta'][:90*16]*0.9977/180*np.pi
    data = data.astype('float32')
theta-=theta[0]
data=np.ascontiguousarray(data.swapaxes(0,1))
theta=np.ascontiguousarray(theta)


# In[4]:


# def remove_stripe_ti(data, beta, mask_size):
#     """Remove stripes with a new method by V. Titareno """
#     gamma = beta*((1-beta)/(1+beta)
#                   )**np.abs(np.fft.fftfreq(data.shape[-1])*data.shape[-1])
#     gamma[0] -= 1
#     v = np.mean(data, axis=0)
#     v = v-v[:, 0:1]
#     v = np.fft.irfft(np.fft.rfft(v)*np.fft.rfft(gamma))
#     mask = np.zeros(v.shape, dtype=v.dtype)
#     mask_size = mask_size*mask.shape[1]
#     mask[:, mask.shape[1]//2-mask_size//2:mask.shape[1]//2+mask_size//2] = 1
#     data[:] += v*mask
#     return data
# data=remove_stripe_ti(data.swapaxes(0,1),0.022,data.shape[-1]).swapaxes(0,1)
data.shape


# Read numpy array with already filtered data

# In[5]:


#data = np.load("foambin2.npy")
[nz, ntheta, n] = data.shape
print([nz,ntheta,n])


# Set the rotation center and projection angles. In this example we have 8 intervals of size $\pi$

# In[6]:


rot_center = 256.0


# Visualization

# In[7]:


def dataplot(slice):
    plt.figure(figsize=(15,8))
    plt.ylabel('x')
    plt.xlabel('theta')    
    plt.imshow(data[slice].swapaxes(0,1),cmap='gray')


# In[8]:


#interact(dataplot,  slice=widgets.IntSlider(min=0, max=data.shape[0]-1,value=data.shape[0]//2));


# The method is solving the problem $\|\mathcal{R}_\text{apr}u-\text{data}\|_2^2+\lambda_0\Big\|\sqrt{\frac{\partial u}{\partial x}+\frac{\partial u}{\partial y}+\frac{\partial u}{\partial z}+\lambda_1\frac{\partial u}{\partial t}}\Big\|_1\to \min$, $\quad$ where $\mathcal{R}_\text{apr}u=\sum_{j=0}^{m-1}\mathcal{R}u_j\varphi_j$

# Init $\lambda_0$ and $\lambda_1$:

# In[9]:


lambda0 = 1.5e-3  # regularization parameter 1
lambda1 = 4  # regularization parameter 2


# The minimization problem is solved by the ADMM scheme with using 'niter' outer ADMM iterations and 'titer' inner tomography iterations. 'titer' in practice should be low.    

# In[10]:


niter = 128  # number of ADMM iterations
titer = 4  # number of inner tomography iterations


# All computations are done on GPUs, where parallelization is done by slices. Variable 'nzp' is the number of slices to process simultaneously by one gpu. 'nzp' is chosen with respect to GPU memory sizes and should be a multiple of 'nz'.     

# In[11]:


nzp = 2 # number of slices to process simultaniously by gpu
ngpus = 1 # number of gpus 


# Take basis functions for decomposition 

# In[12]:


phi = takephi(ntheta) 
m = phi.shape[0] # number of basis functions


# Create a class for reconstruction. The class handles CPU and GPU memory allocation in the C++ context.

# In[13]:


cl = rectv_gpu.Solver(n, ntheta, m, nz, nzp, ngpus)   


# Run reconstruction

# In[14]:


n,ntheta,m,nz,nzp,ngpus,data.shape,data.dtype,theta.shape,theta.dtype


# In[ ]:

rtv = cl.recon(data, theta, phi, rot_center=rot_center,
              lambda0=lambda0, lambda1=lambda1,
              niter=niter, titer=titer)


# Save results as tiff

# In[ ]:


for k in range(rtv.shape[0]):
  dxchange.write_tiff_stack(rtv[k], 'rec_tv/rec_'+str(k), overwrite=True)


# Visualization

# In[ ]:


def foamplot(time,slice):
    plt.figure(figsize=(8,8))
    plt.imshow(rtv[slice,time],vmin=-0,vmax=0.5,cmap='gray')


# In[ ]:


#interact(foamplot,time=widgets.IntSlider(min=0, max=rtv.shape[1]-1, step=1, value=rtv.shape[1]//2),\
#        slice=widgets.IntSlider(min=0, max=rtv.shape[0]-1, step=1, value=rtv.shape[0]//2));

