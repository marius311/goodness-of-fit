module Funcs
export var_of_cls, lda, lda2, pca, blc, test_statistics, test_statistics_max, perturb_lcdm 


function var_of_cls(ell, cls, noisescale = 1.0)
	Ell = repmat(ell,size(cls,1))
	Nell = noisescale * 0.00015048 * exp( Ell.^2 * 7.69112e-7 ) .* (Ell.^2) / 2 / pi
	fsky = 0.3
	2 * (cls + Nell).^2 ./ (2*Ell+1) / fsky
end



function  lda(n_basis, Mat, var_Mat)
	Mat_zerod = Mat .- mean(Mat,1)
	Sigma  = (Mat_zerod.' * Mat_zerod)/size(Mat,1)
	Sigma2 =  diagm(1./mean(var_Mat,1)[:]) * Sigma
	Dtmp2, Vtmp2 = eigs(Sigma2, nev = n_basis)
	D2 = real(Dtmp2)
	Vlda=real(Vtmp2)
	Vlda, D2 
end
function  lda(n_basis, Mat, noisevarvec, cls_lcdm_Mat)
	Mat_zerod = Mat .- mean(Mat,1)
	Sigma  = (Mat_zerod.' * Mat_zerod)/size(Mat,1)

	Mat_zerod_lcdm = cls_lcdm_Mat .- mean(cls_lcdm_Mat,1)
	Sigma_lcdm  = (Mat_zerod_lcdm.' * Mat_zerod_lcdm)/size(cls_lcdm_Mat,1)
	NpS = diagm(noisevarvec[:]) +  Sigma_lcdm
	
	Dtmp2, Vtmp2 = eigs(NpS \ Sigma, nev = n_basis)
	D2 = real(Dtmp2)
	Vlda=real(Vtmp2)
	Vlda, D2 
end
function  lda(n_basis, Mat, noisevarvec, cls_lcdm_Mat, kappa)
	Mat_zerod = Mat .- mean(Mat,1)
	Sigma  = (Mat_zerod.' * Mat_zerod)/size(Mat,1)

	n_basis_proj = n_basis
	VSigma_lcdm, DSigma_lcdm =lda(n_basis_proj, Mat, noisevarvec)
	Sigma_lcdm  = VSigma_lcdm * diagm(DSigma_lcdm[:]) * VSigma_lcdm.'
	NpS = diagm(noisevarvec[:]) +  kappa .* Sigma_lcdm
	
	Dtmp2, Vtmp2 = eigs(NpS \ Sigma, nev = n_basis)
	D2 = real(Dtmp2)
	Vlda=real(Vtmp2)
	Vlda, D2 
end




function  pca(n_basis, Mat)
	mu = mean(Mat,1)
	Mat_zerod = Mat .- mu
	Sigma = (Mat_zerod'*Mat_zerod)/size(Mat,1)
	D1, Vpca = eigs(Sigma, nev = n_basis)
	Vpca, D1
end

function  blc(n_basis, Mat)
	blcsize = int(round(size(Mat,2)/n_basis))
	Vblc = [ones(blcsize),zeros(size(Mat,2)-blcsize)]
	for k = 1:(n_basis-1)
		Vblc = [Vblc circshift(Vblc[:,end],blcsize)]
	end
	Vblc
end

function whiten(V, Mat, var_Mat)
	synth_data = Mat + randn(size(Mat)).*sqrt(var_Mat)
	coeffs = synth_data*V
	covmat = cov(coeffs)
	whitemat = inv(sqrtm(covmat))
	newV = (whitemat * transpose(V)).'
end

function test_statistics(V, synth_data, data)
	synth_coeffs = synth_data*V 
	mean_coeff  = mean(synth_coeffs,1)
	cov_coeff   =  cov(synth_coeffs)
	chisq_data = ( (data*V - mean_coeff) * inv(cov_coeff) * transpose(data*V - mean_coeff) )[1]
	chisq_synth = Array(Float64,0)
	for k=1:size(synth_data,1)
		push!(chisq_synth, ( (synth_data[k,:]*V - mean_coeff) * inv(cov_coeff) * transpose(synth_data[k,:]*V - mean_coeff))[1] )
	end
	chisq_data, chisq_synth
end



function test_statistics_max(Vfull, Dfull, synth_data, data)
	isort = sortperm(1./Dfull)
	Vsort = Vfull[:,isort]
	
	mean_coeff = {}
	inv_coeff = {}
	for maxi=1:size(Vsort,2)
		V = Vsort[:,1:maxi]
		synth_coeffs = synth_data*V
		push!(mean_coeff, mean(synth_coeffs,1))
		cov_coeff   =  cov(synth_coeffs)
		push!(inv_coeff, inv(cov_coeff))
	end

	function getmax(Vsort, mean_coeff, inv_coeff, synth_data, data)
		chisq_max_vec = Array(Float64,0)
		for maxi=1:size(Vsort,2)
			V = Vsort[:,1:maxi]
			chi =  ( (data*V - mean_coeff[maxi]) * inv_coeff[maxi] * transpose(data*V - mean_coeff[maxi]) )[1]
			push!(chisq_max_vec, (chi - maxi)/sqrt(2*maxi) )
		end
		maximum(chisq_max_vec)
	end

	# compute for the data
	chisq_max_data = getmax(Vsort, mean_coeff, inv_coeff, synth_data, data)

	# compute the ensample for the synth data
	chisq_max_synth = Array(Float64,0)
	for k=1:size(synth_data,1)
		push!(chisq_max_synth, getmax(Vsort, mean_coeff, inv_coeff, synth_data, synth_data[k,:]))
	end
	chisq_max_data, chisq_max_synth
end


function rand_get(v)
	v[rand(1:length(v))]
end



function perturb_lcdm(Mat; effectsize = 0.025)
	ell = [0:(size(Mat,2)-1)].'
	Mat_new = Array(Float64,size(Mat))
	sig_sq = [200:400].^2
	mu = [800:100:1200]
	for row = 1:size(Mat_new,1)
		v1 = exp( - 0.5 * abs2(ell- rand_get(mu) ) / rand_get(sig_sq) ).*( sin(ell/20 ) );
		v1 = v1 |> x->(x-mean(x)) |> x-> effectsize*x/(maximum(x) - minimum(x)) |> x-> x+1 
		# subplot(2,1,1)
		# plot((cls_lcdm_Mat[1,:].*v1).')
		# plot((cls_lcdm_Mat[1,:]).')
		# subplot(2,1,2)
		# plot(v1.')
		Mat_new[row,:] = Mat[row,:].*v1 - Mat[row,:]
	end
	Mat_new
end


end