
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
import sys

if __name__ == "__main__":

    in_file = sys.argv[1]
    idF = in_file.find('rect')
    out_file = in_file[:idF-1]+'rotated'+in_file[idF-1:]
    print('rotate',out_file)
    mul=1
    
    data = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=range(0, 256*mul))
    data = data.swapaxes(0,1)
    data = rotate(data, -24.8, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(1,2)
    data = rotate(data, -0.4, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(0,2)
    dxchange.write_tiff_stack(data, out_file, overwrite=True)
    out_file2 = in_file[:idF-1]+in_file[idF:]
    dxchange.write_tiff(data[149*mul,302*mul:775*mul,204*mul:748*mul], 'results/'+out_file2)
