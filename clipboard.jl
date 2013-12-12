
import pypico
import numpy
pico = pypico.load_pico("chain_code/pico3_tailmonty_v33.dat")
# Y_p is set to 4
result = pico.get(**{"scalar_nrun(1)":0,"re_optical_depth":0.085,"theta":0.0104,"omnuh2":0,"omch2":0.125,"ombh2":0.022,"helium_fraction":0.4,"scalar_spectral_index(1)":0.97,"massive_neutrinos":3.046,"scalar_amp(1)":2.5e-9})
numpy.save("data/exotic_cls",result["cl_TT"])
# Y_p is set to 0.248
result = pico.get(**{"scalar_nrun(1)":0,"re_optical_depth":0.085,"theta":0.0104,"omnuh2":0,"omch2":0.125,"ombh2":0.022,"helium_fraction":0.248,"scalar_spectral_index(1)":0.97,"massive_neutrinos":3.046,"scalar_amp(1)":2.5e-9})
numpy.save("data/fiducial_cls",result["cl_TT"])



# import pypico
# import numpy
# pico = pypico.load_pico("chain_code/pico3_tailmonty_v33.dat")
# # "omnuh2":0.0018
# result = pico.get(**{"scalar_nrun(1)":0,"re_optical_depth":0.085,"theta":0.0104,"omnuh2":0.0018,"omch2":0.125,"ombh2":0.022,"helium_fraction":0.248,"scalar_spectral_index(1)":0.97,"massive_neutrinos":3.046,"scalar_amp(1)":2.5e-9})
# numpy.save("data/exotic_cls",result["cl_TT"])

omnuh2_exotic = 0.0018


#####
# see if the data and chain runs line up...
##########


using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

datadir = "data1" 
cls_exotic_Mat = np.load("$datadir/run1_exotic.chain.cls.npy")
data = np.load("$datadir/data_cls_exotic.npy").'
cls_exotic = np.load("$datadir/cls_exotic.npy").'
ell = [0:2500].'

function var_of_cls(ell, cls, noisescale = 1.0)
	Ell = repmat(ell,size(cls,1))
	Nell = noisescale * 0.00015048 * exp( Ell.^2 * 7.69112e-7 ) .* (Ell.^2) / 2 / pi
	fsky = 0.3
	2 * (cls + Nell).^2 ./ (2*Ell+1) / fsky
end
var_cls_exotic = var_of_cls(ell, cls_exotic)


x = ell[:]
y1 = cls_exotic[:]- sqrt(var_cls_exotic[:])
y2 = cls_exotic[:]+ sqrt(var_cls_exotic[:])
plt.fill_between(x, y1, y2, color = "yellow")
plt.plot(x,cls_exotic[:],"--", label= "true exotic Cl")
plt.plot(x,cls_exotic_Mat[200,:][:],"b", label= "A posterior sample exotic Cl")
plt.plot(x,cls_exotic_Mat[1000,:][:],"r", label= "A posterior sample exotic Cl")
plt.plot(x,cls_exotic_Mat[2700,:][:],"g", label= "A posterior sample exotic Cl")
plt.legend()
plt.show()



postmean = mean(cls_exotic_Mat,1)
poststd = std(cls_exotic_Mat,1)
x = ell[:]
y1 =  - poststd[:]
y2 = poststd[:]
plt.plot(x,cls_exotic[:] - postmean[:],"r",label="(posterior mean - true exotic Cl)")
plt.plot(x,0*x,":k")
plt.fill_between(x, y1, y2, facecolor="blue", alpha=0.3 ,label= "posterior 1sigma spread")
#plt.plot(x,cls_exotic[:],"--", label= "true exotic Cl")
plt.title("the blue region shows 1 sigma spread of the posterior")
plt.legend()
plt.show()


#######################################
isort = sortperm(1./Dlda1)
Vlda_sort = Vlda2[:,isort]
Dlda_sort = Dlda2[isort]

isort = sortperm(1./Dpca1)
Vpca_sort = Vpca2[:,isort]
Dpca_sort = Dpca2[isort]


fig = plt.figure()
plt.plot(Vlda_sort[:,1:4], linewidth=1.8)
plt.xlabel("multipole moment", fontsize=16)
plt.title("Top 4 LDA modes", fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("/Users/ethananderes/Desktop/Top4LDA.png",dpi=180)
plt.close(fig)

fig = plt.figure()
plt.plot(Vpca_sort[:,1:4], linewidth=1.8)
plt.xlabel("multipole moment", fontsize=16)
plt.title("Top 4 PCA modes", fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("/Users/ethananderes/Desktop/Top4PCA.png",dpi=180)
plt.close(fig)

fig = plt.figure()
plt.plot(sort(Dpca1, rev=true), linewidth=1.8)
plt.ylabel("eigenvalue spectrum", fontsize=16)
plt.title("Top 20 eigenvalues from PCA", fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("/Users/ethananderes/Desktop/PCAeigen.png",dpi=180)
plt.close(fig)

fig = plt.figure()
plt.plot(sort(Dlda1, rev=true), linewidth=1.8)
plt.ylabel("Generalized eigenvalue spectrum", fontsize=16)
plt.title("Top 20 eigenvalues from LDA", fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("/Users/ethananderes/Desktop/LDAeigen.png",dpi=180)
plt.close(fig)



