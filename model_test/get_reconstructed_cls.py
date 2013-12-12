import sys
from model_test_utils import *
from pylab import *

filename= sys.argv[1]
filename = str(filename)
cls_init = np.load(filename)
cls_init = cls_init[:,:2500]
cl_fid = np.load('clfid.npy')[:2500]
modes = get_modes()
plot(modes[0,1,:])
plot(modes[0,13,:])
plot(modes[1,13,:])
show()
cls_rec = cls_init.copy()
for i in arange(cls_init[:,0].size - 1):
	cls_rec[i,:] = reconstruct_cl(cls_init[i,:], cl_fid, modes)

np.save(filename+'reconstructed.npy', cls_rec)
