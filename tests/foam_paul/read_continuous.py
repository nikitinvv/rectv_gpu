#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import h5py
import tomopy
import dxchange

h5fname = "/data/staff/tomograms/viknik/foam/149_ASM_SP_1ktps.h5"
sino = (32, 96) # slices for reconstructions
nframes = 8 # time frames for reconstruction
frame = (1487780)//40 # middle time frame for reconstruction
nproj = 40 # number of angles for 180 degrees interval
binning = 0 

proj, flat, dark, theta = dxchange.read_aps_32id(h5fname, sino=sino, proj=((frame-nframes//2)*nproj,(frame+nframes//2)*nproj))
theta = theta[(frame-nframes//2)*nproj:(frame+nframes//2)*nproj]
print(proj.shape)
# Flat-field correction of raw data.
data = tomopy.normalize(proj, flat, dark, cutoff=1.4)

# remove stripes
data = tomopy.remove_stripe_fw(
    data, level=7, wname='sym16', sigma=1, pad=True)

# log filter
#data = tomopy.minus_log(data)
data = tomopy.remove_nan(data, val=0.0)
data = tomopy.remove_neg(data, val=0.00)
data[np.where(data == np.inf)] = 0.00
data = tomopy.prep.phase.retrieve_phase(data, pixel_size=0.00025, dist=50, energy=20, alpha=0.015, pad=True)
# reshape for 4d
data = np.reshape(data,[nframes*nproj,data.shape[1],data.shape[2]])
np.save('foam',data.swapaxes(0,1))
