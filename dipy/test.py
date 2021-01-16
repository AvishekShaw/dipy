import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.custom import (gaussian_noisifier,add_zero_noise)
from scipy.sparse.linalg import cg as conj_grad

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi') # Data shape (81,106,76,160). 160 no. of directions
data, affine = load_nifti(hardi_fname)

#Control variables
noise_type = None # Select out of None, 'gaussian', 'zero_noise'
percent_noise_list = [20,40,60,80]
fraction_noisy_voxels  = 0 # Wont run if noise_type is None
data_small = data[15:46, 45:76, 16:47]
csd_fit = False # runs the CSD fitting procedure
read_data = True # to read files, code segment below
algo_list=['cholesky','qr','svd']
algo = 'cholesky'
num_csd_loops = 1 
plot_data = False

# There are 160 bvals corresponding to 160 directions and bvecs is an array of (160,3) corresponding to 
# the direction vectors of each bval.
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.0)
nvoxels = np.sum(mask) # nvoxels: no. of voxels used for calc. of response function
response, ratio = response_from_mask_ssst(gtab, data, mask)

if csd_fit:
	from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
	# for fraction_noisy_voxels in [0.2,0.4,0.6,0.8]:
	csd_model = ConstrainedSphericalDeconvModel(gtab, response, noise_type,fraction_noisy_voxels,algo)
	for _ in range(num_csd_loops):
		start = time.time()
		csd_fit = csd_model.fit(data_small)
		end=time.time()
		print("The time taken for CSD fit in secs:"+str(end-start),"\n")

'''Following code fragment is for finding out the norm difference of fODF arrays produced
by different techniques. For eg: you could find out the fODF value by cholesky with no noise
and numpy least squares with 20 percent noise and find out the difference between them and plot
them to see the perturbation effect of induced noise.'''
if read_data:

	cholesky_fodf = pd.read_csv("./cholesky_None_0.csv").values
	for algorithm in algo_list:
		for percent_noise in percent_noise_list:
			filename = str(algorithm)+"_gaussian"+"_"+str(percent_noise)+".csv"
			print(filename)
			other_fodf= pd.read_csv(filename).values
			residual_norm = np.linalg.norm((cholesky_fodf-other_fodf),ord=2,axis=1)
			cholesky_norm = np.linalg.norm((cholesky_fodf),ord=2,axis=1)
			norm_percent = (residual_norm/cholesky_norm)*100

			if plot_data:
				_ = plt.hist(norm_percent,bins='auto')
				plt.xlabel("differece(%)", fontsize = 12)
				plt.ylabel("No. of voxels", fontsize = 12)
				plt.title("Histogram", fontsize = 12)
				plotname = str(algorithm)+"_gaussian"+"_"+str(percent_noise)+".jpg"
				plt.savefig(plotname,dpi=300)
			print("Method : %s, noise percent : %f, norm difference percent: %f" %(algorithm, percent_noise,np.mean(norm_percent)))

# A = np.random.randn(3,3)
# print(A)
# print(gaussian_noisifier(A,0.8))