import tomopy
import dxchange
import numpy as np
import h5py
import sys
import skimage.feature

##################################### Inputs #########################################################################
file_name = '/local/data/vnikitin/vincent/alignment/Chip_interlaced_9100eV_2000prj_10cycles_2s_069.h5'
ndsets = 8
sino_start = (512-128+50)*2
sino_end = (512+128+50)*2
theta_start = 0
theta_end = 200
flat_field_norm = True
flat_field_drift_corr = True  # Correct the intensity drift
remove_rings = True
binning = 1
######################################################################################################################


def preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings, FF_drift_corr=flat_field_drift_corr, downsapling=binning):

    if FF_norm:  # dark-flat field correction
        prj = tomopy.normalize(prj, flat, dark)
    if FF_drift_corr:  # flat field drift correction
        prj = tomopy.normalize_bg(prj, air=100)
    prj[prj <= 0] = 1  # check dark<data
    prj = tomopy.minus_log(prj)  # -logarithm
    if remove_rings:  # remove rings
        prj = tomopy.remove_stripe_fw(
            prj, level=7, wname='sym16', sigma=1, pad=True)
    if downsapling > 0:  # binning
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
    return prj


if __name__ == "__main__":
   # read data
    prj, flat, dark, theta = dxchange.read_aps_32id(
        file_name, sino=(sino_start, sino_end), proj=(theta_start,theta_end*ndsets))
    # preprocess
    prj = preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings,
                          FF_drift_corr=flat_field_drift_corr, downsapling=binning)

    prj=prj[:,:,456//pow(2,binning):-456//pow(2,binning)]
    print(np.linalg.norm(prj))
    np.save('prjbin1',prj)        
    np.save('theta',theta)  
        