"""Module for 4D tomography."""

import numpy as np
from rectv_gpu.rectv import rectv


def getp(a):
    return a.__array_interface__['data'][0]


class Solver(rectv):
    """Base class for the time-resolved tomography sovler.
    Attributes
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    nzp : int
        The number of slices to process together
        simultaneously by each GPU
    ngpus : int
        Number of gpus
    rot_center : int
        Rotation center in tomography
    lambda0 : float
        Regularization parameter 1, Default: 1e-3
    lambda1 : int
        Regularization parameter 2, Default: 1 
    step : float
        Gradient step size, adapted during iterations, Default: 1
    """

    def __init__(self, n, ntheta, m, nz, nzp, ngpus):
        """Please see help(Solver) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(n, ntheta, m, nz, nzp, ngpus)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def recon(self, data, theta, phi, rot_center=None,
              lambda0=1e-3, lambda1=1, step=0.5,
              niter=1, titer=1, dbg=False):
        """Reconstruct with time-resolved method"""
        if(rot_center == None):
            rot_center = self.n/2
        rtv = np.zeros([self.nz, self.m, self.n, self.n],
                       dtype='float32')  # memory for result
        print(lambda0)
        self.run(getp(rtv), getp(data), getp(
            theta), getp(phi), rot_center, lambda0, lambda1, step, niter, titer, dbg)
        return rtv

    def adjoint_tests(self, data, theta, phi, lambda1, rot_center=None):
        """Test for the adjoint operator"""
        if(rot_center == None):
            rot_center = self.n/2
        res = np.zeros([6])
        self.check_adjoints(getp(res),getp(data), getp(theta), getp(phi), lambda1, rot_center)
        return res
