
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
import sys

if __name__ == "__main__":

    in_file = sys.argv[1]
    idF = in_file.find('rec')+5
    out_file = 'rotated'+in_file[idF-1:]
    #print('rotate',out_file)
    mul=1
    data = np.zeros([256,768,768],dtype='float32')
    print(in_file+'/r_00000.tiff')
    
    data0 = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=range(0, 256-64))
    data[32:-32]=data0
    data = np.rot90(data,1,axes=(1,2))
    data = data.swapaxes(0,1)
    data = rotate(data, -24.8, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(1,2)
    data = rotate(data, -0.4, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(0,2)
    dxchange.write_tiff_stack(data, 'rotated/'+out_file+'/r', overwrite=True)
   # out_file2 = 'r'#in_file[:idF-1]+in_file[idF:]
    dxchange.write_tiff(data[161*mul,302*mul-100:775*mul-100,204*mul-100:748*mul-100], 'results/'+out_file)
