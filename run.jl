
using PyCall, Funcs
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
datadir = "data2" 

###################
# 
#  load data and lcdm runs
#
#################
data = np.load("$datadir/data_cls_exotic.npy")[:].'
ell = [0:(length(data)-1)].'

cls_exotic_Mat = np.load("$datadir/run1_smoothspectra.chain.cls.npy")
cls_lcdm_Mat   = np.load("$datadir/run1_lcdm.chain.cls.npy")
cls_lcdm_Matcut1 = cls_lcdm_Mat[1:round(end/2),:]
cls_lcdm_Matcut2 = cls_lcdm_Mat[(round(end/2)+1):end,:] #this one is used to determine the sampling distribution of 
noisevar_lcdm_Mat = var_of_cls(ell, cls_lcdm_Mat)
noisevar_exotic_Mat = var_of_cls(ell, cls_exotic_Mat)
noisevar_lcdm_Matcut1 = noisevar_lcdm_Mat[1:round(end/2),:]
noisevar_lcdm_Matcut2 = noisevar_lcdm_Mat[(round(end/2)+1):end,:]


###################
# 
#  compute the detection modes 
#
#################
n_basis = 25
#Vlda2, Dlda2 =  lda(n_basis, cls_exotic_Mat, mean(noisevar_lcdm_Matcut1,1) + var(cls_lcdm_Matcut1,1)) # diag version
Vlda2, Dlda2 =  lda(n_basis, cls_exotic_Mat, mean(noisevar_lcdm_Matcut1,1), cls_lcdm_Matcut1) # full cov version
#Vlda2, Dlda2 =  lda(n_basis, cls_exotic_Mat, mean(noisevar_lcdm_Matcut1,1), cls_lcdm_Matcut1, 1.0) # low rank version
Vpca2, Dpca2 =  pca(n_basis, cls_exotic_Mat)
Vlda1, Dlda1 =  lda(n_basis, cls_lcdm_Matcut1, mean(noisevar_lcdm_Matcut1,1)) # low rank version
Vpca1, Dpca1 =  pca(n_basis, cls_lcdm_Matcut1)
Vblc2 = blc(n_basis, cls_exotic_Mat) 
# plt.plot(Vlda2);plt.show()


###################
# 
# compare with what we would expect if the model was correct.
#
#################
synth_data = cls_lcdm_Matcut2 + randn(size(cls_lcdm_Matcut2)).*sqrt(noisevar_lcdm_Matcut2)
chisq_lda_data2, chisq_lda_synth2 = test_statistics(Vlda2, synth_data, data)
chisq_pca_data2, chisq_pca_synth2 = test_statistics(Vpca2, synth_data, data)
chisq_lda_data1, chisq_lda_synth1 = test_statistics(Vlda1, synth_data, data)
chisq_pca_data1, chisq_pca_synth1 = test_statistics(Vpca1, synth_data, data)
chisq_blc_data2, chisq_blc_synth2 = test_statistics(Vblc2, synth_data, data)
z_lda_data2, z_lda_synth2 = test_statistics_max(Vlda2, Dlda2, synth_data, data)
z_pca_data2, z_pca_synth2 = test_statistics_max(Vpca2, Dpca2, synth_data, data)
z_lda_data1, z_lda_synth1 = test_statistics_max(Vlda1, Dlda1, synth_data, data)
z_pca_data1, z_pca_synth1 = test_statistics_max(Vpca1, Dpca1, synth_data, data)




############################
#
# plot the results and print out the quasi p-values
#
###########################
using Distributions
chisqX = Chisq(n_basis)

println("""
chisq_lda2 = $chisq_lda_data2, z-score = $((chisq_lda_data2 - mean(chisq_lda_synth2))/std(chisq_lda_synth2))
chisq_pca2 = $chisq_pca_data2, z-score = $((chisq_pca_data2 - mean(chisq_pca_synth2))/std(chisq_pca_synth2))

chisq_lda1 = $chisq_lda_data1, z-score = $((chisq_lda_data1 - mean(chisq_lda_synth1))/std(chisq_lda_synth1))
chisq_pca1 = $chisq_pca_data1, z-score = $((chisq_pca_data1 - mean(chisq_pca_synth1))/std(chisq_pca_synth1))
chisq_blc2 = $chisq_blc_data2, z-score = $((chisq_blc_data2 - mean(chisq_blc_synth2))/std(chisq_blc_synth2))

z_lda2 = $z_lda_data2, z-score = $((z_lda_data2 - mean(z_lda_synth2))/std(z_lda_synth2))
z_pca2 = $z_pca_data2, z-score = $((z_pca_data2 - mean(z_pca_synth2))/std(z_pca_synth2))

z_lda1 = $z_lda_data1, z-score = $((z_lda_data1 - mean(z_lda_synth1))/std(z_lda_synth1))
z_pca1 = $z_pca_data1, z-score = $((z_pca_data1 - mean(z_pca_synth1))/std(z_pca_synth1))
""")



#######################
#
#  here are what the exotic perturbation model looks like
#
######################
savetofile = true

cls_exotic = np.load("$datadir/cls_exotic.npy").'
cls_bestfit_lcdm = mean(cls_lcdm_Mat,1)
delt_cls_exotic = cls_exotic - cls_bestfit_lcdm


fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(cls_exotic.', linewidth=1.8, label="non LCDM simulation truth")
plt.plot(cls_bestfit_lcdm.', linewidth=1.8, label="best fit LCDM")
plt.tick_params(labelsize=14)
plt.legend()
plt.subplot(2,1,2)
plt.plot(delt_cls_exotic.', linewidth=1.8, label="truth - best fit LCDM")
plt.tick_params(labelsize=14)
plt.legend(loc=4)
savetofile ? (plt.savefig("/home/anderes/Dropbox/Lloyd_Brent_LDM/Exotic_vrs_lcdm.png",dpi=180); plt.close(fig)) : plt.show()


howmanymodes = 4

isort2 = sortperm(1./Dlda2)
Vlda_sort2 = Vlda2[:,isort2]
Dlda_sort2 = Dlda2[isort2]
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(Vlda_sort2[:,1:howmanymodes], linewidth=1.8)
plt.xlabel("multipole moment", fontsize=14)
plt.title("Top $howmanymodes LDA modes projecting out LDA on LCDM", fontsize=12)
plt.tick_params(labelsize=14)
plt.subplot(1,2,2)
plt.semilogy(sort(Dlda2, rev=true), linewidth=1.8)
plt.title("Generalized eigenvalue spectrum", fontsize=14)
plt.tick_params(labelsize=14)
savetofile ? (plt.savefig("/home/anderes/Dropbox/Lloyd_Brent_LDM/LDA_prjoutLCDM_$howmanymodes.png",dpi=180); plt.close(fig)) : plt.show()



isort1 = sortperm(1./Dlda1)
Vlda_sort1 = Vlda1[:,isort1]
Dlda_sort1 = Dlda1[isort1]
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(Vlda_sort1[:,1:howmanymodes], linewidth=1.8)
plt.xlabel("multipole moment", fontsize=14)
plt.title("Top $howmanymodes LDA on LCDM modes", fontsize=14)
plt.tick_params(labelsize=14)
plt.subplot(1,2,2)
plt.semilogy(sort(Dlda1, rev=true), linewidth=1.8)
plt.title("Generalized eigenvalue spectrum", fontsize=14)
plt.tick_params(labelsize=14)
savetofile ? (plt.savefig("/home/anderes/Dropbox/Lloyd_Brent_LDM/LDAonLCDM_$howmanymodes.png",dpi=180); plt.close(fig)) : plt.show()


isort = sortperm(1./Dpca1)
Vpca_sort2 = Vpca1[:,isort]
Dpca_sort2 = Dpca1[isort]
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(Vpca_sort2[:,1:howmanymodes], linewidth=1.8)
plt.xlabel("multipole moment", fontsize=14)
plt.title("Top $howmanymodes PCA modes", fontsize=14)
plt.tick_params(labelsize=14)
plt.subplot(1,2,2)
plt.semilogy(sort(Dpca1, rev=true), linewidth=1.8,label="eigenvalues")
plt.title("Eigenvalue spectrum", fontsize=14)
pca_StoN = var(cls_exotic_Mat*Vpca_sort2,1)./ var( (randn(size(cls_exotic_Mat)).*sqrt(noisevar_exotic_Mat))*Vpca_sort2,1)
plt.semilogy(pca_StoN[:], linewidth=1.8,label="signal to noise ratio")
plt.tick_params(labelsize=14)
plt.legend()
savetofile ? (plt.savefig("/home/anderes/Dropbox/Lloyd_Brent_LDM/PCAonExotic_$howmanymodes.png",dpi=180); plt.close(fig)) : plt.show()


fig = plt.figure()
lda_StoN = var(cls_exotic_Mat*Vpca2,1)./ var( (randn(size(cls_exotic_Mat)).*sqrt(noisevar_exotic_Mat))*Vpca2,1)
plt.semilogy(sort(lda_StoN[:], rev=true), linewidth=1.8, label="LDA S/N")
plt.semilogy(sort(pca_StoN[:], rev=true), linewidth=1.8,label="PCA S/N")
plt.legend()
savetofile ? (plt.savefig("/home/anderes/Dropbox/Lloyd_Brent_LDM/StoNs.png",dpi=180); plt.close(fig)) : plt.show()


##################
# Lets verify what Dlda tell you

Dlda2test = var(cls_exotic_Mat*Vlda2,1)./ var(( cls_lcdm_Matcut1 + randn(size(cls_lcdm_Matcut1) ).*sqrt(noisevar_lcdm_Matcut1) )*Vlda2,1)
print([Dlda2test.' Dlda2])
print((Dlda2test.'- Dlda2)./Dlda2)


Dlda1test = var(cls_lcdm_Matcut1*Vlda1,1)./ var( (randn(size(cls_lcdm_Matcut1)).*sqrt(noisevar_lcdm_Matcut1))*Vlda1,1)
print([Dlda1test.' Dlda1])
print((Dlda1test.'- Dlda1)./Dlda1)




# x = linspace(0,max(max(chisq_lda_synth),max(chisq_pca_synth),max(chisq_blc_synth)),100)
# plt.subplot(1,3,1)
# plt.hist(chisq_lda_synth,30,normed=1)
# plt.plot(x,pdf(chisqX,x))
# plt.plot(chisq_lda_data,0,"r*",markersize = 20)
# plt.subplot(1,3,2)
# plt.hist(chisq_pca_synth,30,normed=1)
# plt.plot(x,pdf(chisqX,x))
# plt.plot(chisq_pca_data,0,"r*",markersize = 20)
# plt.subplot(1,3,3)
# plt.hist(chisq_blc_synth,30,normed=1)
# plt.plot(x,pdf(chisqX,x))
# plt.plot(chisq_blc_data,0,"r*",markersize = 20)
# plt.show()


