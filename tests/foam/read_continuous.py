#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct a single data set.
"""

import numpy as np
import h5py
import tomopy
import dxchange

h5fname = "/data/staff/tomograms/viknik/rajmund/dk_MCFG_1_p_s1_.h5"
sino = (1300, 1364) # slices for reconstructions
nframes = 8 # time frames for reconstruction
frame = 94 # middle time frame for reconstruction
nproj = 300 # number of angles for 180 degrees interval
binning = 1 

proj, flat, dark, theta = dxchange.read_aps_32id(h5fname, sino=sino, proj=((frame-nframes//2)*nproj,(frame+nframes//2)*nproj))
theta = theta[(frame-nframes//2)*nproj:(frame+nframes//2)*nproj]
print(proj.shape)
print(theta.shape)

# Flat-field correction of raw data.
data = tomopy.normalize(proj, flat, dark, cutoff=1.4)

# remove stripes
data = tomopy.remove_stripe_fw(
    data, level=7, wname='sym16', sigma=1, pad=True)

# log filter
data = tomopy.minus_log(data)
data = tomopy.remove_nan(data, val=0.0)
data = tomopy.remove_neg(data, val=0.00)
data[np.where(data == np.inf)] = 0.00

# Binning
data = tomopy.downsample(data, level=binning, axis=2)
if data.shape[1] > 1:
    data = tomopy.downsample(data, level=binning, axis=1)

# reshape for 4d
data = np.reshape(data,[nframes*nproj,data.shape[1],data.shape[2]])
np.save('foambin2.npy',data.swapaxes(0,1))
