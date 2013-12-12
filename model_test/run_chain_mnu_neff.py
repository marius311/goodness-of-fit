#!/usr/bin/env python2.7

"""
Runs a LCDM chain on a fake Gaussian Planck-like likelihood. 

Usage:
    run_chain.py DL_DATA OUTPUT

DL_DATA - File which contains data Cl's. 
OUTPUT - Output file root. 

DL and DL_DATA should be text files with two columns, corresponding to l, Dl, starting at l=0.
"""

import sys
sys.path.append('../julia_code/cosmoslik')
from cosmoslik import param_shortcut, lsum, get_plugin, SlikDict, SlikPlugin, Slik, mpi
from cosmoslik.chains import load_chain
from numpy import identity, exp, inf, hstack, load, zeros, arange, pi, sqrt, loadtxt, save, array
from numpy.random import normal

        
param = param_shortcut('start','scale')

class main(SlikPlugin):
    
    def __init__(self):
        super(SlikPlugin,self).__init__()
        
        #
        # The Cl, Nl, fsky, and lranges used in the likelihood
        #
        self.nl_fid = 0.00015048 * exp( arange(2501)**2 * 7.69112e-7 ) * (arange(2501)**2/2/pi)
        self.fsky = 0.3
        self.lrange = (2,2501)
        self.lslice = slice(*self.lrange)
        self.ells = arange(*self.lrange)
        #

        #
        # The data Cl
        #
        self.cl_data = load(sys.argv[1])[self.lslice]
        #


        #Setting up CosmoSlik stuff
   
        self.cosmo = get_plugin('models.cosmology')(
            logA = param(3.2),
            ns = param(0.96),
            ombh2 = param(0.0221),
            omch2 = param(0.12),
            tau = param(0.09,min=0,gaussian_prior=(0.0925,0.015)),
            theta = param(0.010413),
            omnuh2 = param(0.000645, scale = .00005),
            massive_neutrinos = param(3.04, scale = .1)
	)
	print self.cosmo
                       
        self.get_cmb = get_plugin('models.pico')(
            datafile='chain_code/pico3_tailmonty_v33.dat'
        )

        self.bbn = get_plugin('models.bbn_consistency')()
        self.hubble_theta = get_plugin('models.hubble_theta')()  
        self.priors = get_plugin('likelihoods.priors')(self)
    
        self.sampler = get_plugin('samplers.metropolis_hastings')(
             self,
	     num_samples=50000,
             output_file=sys.argv[2]+'.chain',
             proposal_cov='chain_code/proposal.covmat',
             proposal_scale=1,
             output_extra_params=['cosmo.H0','cosmo.Yp','cl_TT']
        )
    
    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10
        self.cosmo.Yp = self.bbn(**self.cosmo)
        self.cosmo.H0 = self.hubble_theta.theta_to_hubble(**self.cosmo)
        
        self.cmb_result = self.get_cmb(outputs=['cl_TT'],force=True,**self.cosmo)
        self.cl_TT = {'cl_TT':self.cmb_result['cl_TT'][0:2501]}
      
        self.varCl = (2*(self.cmb_result['cl_TT'][0:2501] + self.nl_fid)**2/self.fsky)[self.lslice]/(2*self.ells+1)
      
        dx = self.cmb_result['cl_TT'][self.lslice] - self.cl_data
        return sum(dx**2 / self.varCl / 2) + self.priors(self)


if __name__=='__main__':
    #run the chain
    for _ in Slik(main()).sample(): pass
