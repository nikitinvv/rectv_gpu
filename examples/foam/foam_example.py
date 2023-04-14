#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rectv_gpu
import numpy as np
import dxchange


# Initiate basis functions for function decomposition $u=\sum_{j=0}^{m-1}u_j\varphi_j$

# In[2]:


def takephi(ntheta):
    m = 8  # number of basis functions
    [x, y] = np.meshgrid(np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
    phi = np.zeros([m, 2*ntheta], dtype='float32')
    phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
    return phi


# Read numpy array with already filtered data

# In[3]:


data = np.load("foambin2.npy")
[nz, ntheta, n] = data.shape
print([nz,ntheta,n])


# Set the rotation center and projection angles. In this example we have 8 intervals of size $\pi$

# In[4]:


rot_center = n//2 
theta = np.linspace(0, 8*np.pi, ntheta, endpoint=False).astype('float32')  


# The method is solving the problem $\|\mathcal{R}_\text{apr}u-\text{data}\|_2^2+\lambda_0\Big\|\sqrt{\frac{\partial u}{\partial x}+\frac{\partial u}{\partial y}+\frac{\partial u}{\partial z}+\lambda_1\frac{\partial u}{\partial t}}\Big\|_1\to \min$, $\quad$ where $\mathcal{R}_\text{apr}u=\sum_{j=0}^{m-1}\mathcal{R}u_j\varphi_j$

# Init $\lambda_0$ and $\lambda_1$:

# In[18]:


lambda0 = 1e-2  # regularization parameter 1
lambda1 = 256  # regularization parameter 2

# The minimization problem is solved by the ADMM scheme with using 'niter' outer ADMM iterations and 'titer' inner tomography iterations. 'titer' in practice should be low.    

# In[6]:


niter = 16  # number of ADMM iterations
titer = 4  # number of inner tomography iterations


# All computations are done on GPUs, where parallelization is done by slices. Variable 'nsp' is the number of slices to process simultaneously by one gpu. 'nzp' is chosen with respect to GPU memory sizes and should be a multiple of 'nz'.     

# In[7]:


nzp = 8 # number of slices to process simultaniously by gpu
ngpus = 1 # number of gpus 


# Take basis functions for decomosition 

# In[8]:


phi = takephi(ntheta) 
m = phi.shape[0] # number of basis functions


# Create a class for reconstruction. The class handles CPU and GPU memory allocation in the C++ context.

# In[9]:


cl = rectv_gpu.Solver(n, ntheta, m, nz, nzp, ngpus)   


# In[19]:


rtv = cl.recon(data, theta, phi, rot_center=rot_center,
              lambda0=lambda0, lambda1=lambda1,
              niter=niter, titer=titer)


# Save results as tiff

# In[20]:


for k in range(rtv.shape[0]):
   dxchange.write_tiff_stack(rtv[k], 'rec_tv/rec_'+str(k), overwrite=True)


# In[ ]:




