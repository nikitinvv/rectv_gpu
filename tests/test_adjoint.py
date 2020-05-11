
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import rectv_gpu


class TestRectv(unittest.TestCase):
    """Test rectv methods for consistency."""

    def setUp(self, n=256, ntheta=128, m=16, nz=1, nzp=1, ngpus=1, lambda1=4):
        self.n = n
        self.ntheta = ntheta
        self.nz = nz
        self.nzp = nzp
        self.m = m
        self.ngpus = ngpus
        self.lambda1 = lambda1

        [x, y] = np.meshgrid(
            np.arange(-ntheta//2, ntheta//2), np.arange(-m//2, m//2))
        self.phi = np.zeros([m, 2*ntheta], dtype='float32')
        self.phi[:, ::2] = np.cos(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
        self.phi[:, 1::2] = np.sin(2*np.pi*x*y/ntheta)/np.sqrt(ntheta)
        
        print(rectv_gpu)

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        np.random.seed(0)
        data = np.random.random([self.nz, self.ntheta, self.n]).astype('float32')
        theta = np.linspace(0, 8*np.pi, self.ntheta).astype('float32')
        
        with rectv_gpu.Solver(self.n, self.ntheta, self.m, self.nz, self.nzp, self.ngpus) as slv:
            res = slv.adjoint_tests(data, theta, self.phi, self.lambda1)
            print('<R*data, R*data> = {:.6f}'.format(
                res[0]))
            print('<data, RR*data> = {:.6f}'.format(
                res[1]))
            print('<RR*data, RR*data> = {:.6f}'.format(
                res[2]))
            
            print('<G*data, G*data> = {:.6f}'.format(
                res[3]))
            print('<data, GG*data> = {:.6f}'.format(
                res[4]))
            print('<GG*data, GG*data> = {:.6f}'.format(
                res[5]))
            
            # Test whether Adjoint is correct
            np.testing.assert_allclose(res[0],res[1], rtol=1e-1)
            np.testing.assert_allclose(res[1],res[2], rtol=1e-1)
            np.testing.assert_allclose(res[3],res[4], rtol=1e-1)
            np.testing.assert_allclose(res[4],res[5], rtol=1e-1)
            
if __name__ == '__main__':
    unittest.main()