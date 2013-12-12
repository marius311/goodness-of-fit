# ssh -Y anderes@physauth.physics.ucdavis.edu
# RIrmwm3?DMzJzYBo
# ssh -Y virgo01
# dropbox start

using PyCall, Funcs
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
datadir = "data2" # you need to creat this directory if it doesn't already exist...

#####################
# 
# generate the nonparametric model space which realistic model departures/fluctuations
#
####################################
#  generate a standard model 
pythonscript1 = "import pypico; import numpy
pico = pypico.load_pico('chain_code/pico3_tailmonty_v33.dat')
result = pico.get(**{'scalar_nrun(1)':0,'re_optical_depth':0.085,'theta':0.0104,'omnuh2':0.000645,'omch2':0.125,'ombh2':0.022,'helium_fraction':0.248,'scalar_spectral_index(1)':0.97,'massive_neutrinos':3.046,'scalar_amp(1)':2.5e-9})
numpy.save('$datadir/temp_cls',result['cl_TT'])" |> str->replace(str,'\n',';')
run(`python -c $pythonscript1`)
### get reasonable perturbations aroudn that model
temp_cls = np.load("$datadir/temp_cls.npy")[1:2501].'
np.save("$datadir/temp_cls", temp_cls[:]) # re-write over temp_cls.npy
ScaleNoise = 10
NumRuns = 5000000
run(`mpiexec -n 5 chain_code/run_chain_smoothspectra.py  $datadir/temp_cls.npy $datadir/run1_smoothspectra $ScaleNoise $NumRuns`)
run(`./extract_cls.py $datadir/run1_smoothspectra.chain`)
###  now grab one of the nonparametric models and use it as data
all_exotic = np.load("$datadir/run1_smoothspectra.chain.cls.npy")
cls_exotic = all_exotic[randperm(size(all_exotic,1))[1],:]
ell = [0:2500].'
data_cls_exotic =  cls_exotic + randn(size(ell)).*sqrt(var_of_cls(ell, cls_exotic)) # plt.plot(delt_cls_exotic.');plt.show()
np.save("$datadir/cls_exotic", cls_exotic[:])
np.save("$datadir/data_cls_exotic", data_cls_exotic[:])
#run(`rm $datadir/run1_smoothspectra.chain*`)


#########################################
#
# sample from the posterior under LCDM for the modes that we want to project out
#
#########################################
run(`mpiexec -n 5 chain_code/run_chain_lcdm.py  $datadir/data_cls_exotic.npy $datadir/run1_lcdm`)
run(`./extract_cls.py $datadir/run1_lcdm.chain`)


########################################
#
# sample from exotic model: I'm just not sure we want to fit this to data...wont the highly constrained exotic model fluxtuations have low signal
# in this posterior.
#
######################################### 
ScaleNoise = 1
NumRuns = 5000
run(`mpiexec -n 5 chain_code/run_chain_smoothspectra.py  $datadir/data_cls_exotic.npy $datadir/run1_smoothspectra2 $ScaleNoise $NumRuns`)
run(`./extract_cls.py $datadir/run1_smoothspectra2.chain`)








