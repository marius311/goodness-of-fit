#from pylab import * 
from numpy import sqrt, array, dot, arange, cos, sin, zeros
def get_modes(maxell = 2500, maxn = 26):
	#clfid = array(clfid)
	pi = 3.141592654
	N = 2 * pi/ (maxell)
	modes = zeros([2, maxn, maxell])
	for i in arange(maxn-1):
		for j in arange(maxell-1):
			modes[0,i,j] = cos(N * i * (j))#*clfid[j]
			modes[1,i,j] = sin(N * i * (j))
		norm = sqrt(dot(modes[0,i,:], modes[0,i,:]))
		modes[0,i,:] = modes[0,i,:]/norm#/sqrt(dot(modes[0,i,:],modes[0,i,:]))
		norm = sqrt(dot(modes[1,i,:], modes[1,i,:]))
		if norm != 0:
			modes[1,i,:] = modes[1,i,:]/norm#/sqrt(dot(modes[1,i,:],modes[1,i,:]))
		print dot(modes[1,i,:], modes[1,i,:]), dot(modes[1,i,:], modes[0,i,:])
	return modes

def reconstruct_cl(cl, clfid, modes, maxell = 10000):
	cl = array(cl)
	clfid = array(clfid)
	cl_temp = cl/clfid
	if (cl.size > maxell): cl = cl[:maxell]
	if (cl.size > modes[0,0,:].size): cl = cl[:modes[0,0,:].size]
	n = cl.size
	cl_reconstructed = zeros(n)
	for i in arange(modes[0,:,0].size -1):
		amp0 = dot(cl_temp, modes[0,i,:n])
		amp1 = dot(cl_temp, modes[1,i,:n])
		cl_reconstructed = cl_reconstructed + amp0 * modes[0,i,:n] + amp1 * modes[1,i,:n]
	cl_reconstructed = cl_reconstructed * clfid
	return cl_reconstructed

#def get_cl(cosmology, maxell = 2500):
	
