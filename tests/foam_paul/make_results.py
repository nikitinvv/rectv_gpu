import numpy as np
import dxchange
import matplotlib.pyplot as plt
lambda0a = [1e-3]  # regularization parameter 1
lambda1a = [8,16,32,64]  # regularization parameter 2
ma = [8,16,32]
cnt = 0
a = np.zeros([200,200*8])
for i in range(len(ma)):
    for k in range(len(lambda0a)):
        #for j in range(len(lambda1a)):
            m = ma[i]
            lambda0=lambda0a[k]
            lambda1=m#lambda1a[j]
            import os
            name0 = 'figs'
            if not os.path.exists(name0):
                os.makedirs(name0)
            for kk in range(ma[i]):
                name = 'resrec_tv'+str(m)+str(lambda0)+str(lambda1)+'/rec_'+str(kk)+'__'+'00064'
                #print(name)
                a[:,np.mod(cnt,8)*200:np.mod(cnt,8)*200+200]=dxchange.read_tiff(name+'.tiff')[280:280+200,155:155+200]     
                cnt+=1                           
                if(np.mod(cnt,8)==0 and cnt>0):                
                    vmin = np.mean(a)-np.std(a)*2
                    vmax = np.mean(a)+np.std(a)*2
                    name = 'rec_tv'+str(m)+str(lambda0)+str(lambda1)+'_'+str(cnt)
                    plt.imsave('figs/'+name+'.png',a,vmin=vmin,vmax=vmax,cmap='gray')
                    print('\\includegraphics[width=0.1\\textwidth]{{figs/%s.png}} \\\\ ' % (name)),
                    # exit()
                
                    
                # dxchange.write_tiff_stack(np.rot90(rtv[32],1,axes=(1,2))[:,200:350,200:450], 'rec_tvtime'+str(m)+str(lambda0)+str(lambda1)+'/rec.tiff', overwrite=True)
                #dxchange.write_tiff(np.rot90(rtv[16,7],1,axes=(1,2)), 'rec_tvtimenew/'+str(lambda0)+str(lambda1)+'.tiff', overwrite=True)
            