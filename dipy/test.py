import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import csv

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.custom import (gaussian_noisifier,add_zero_noise)
from scipy.sparse.linalg import cg as conj_grad


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi') # Data shape (81,106,76,160). 160 no. of directions
data, affine = load_nifti(hardi_fname)
# There are 160 bvals corresponding to 160 directions and bvecs is an array of (160,3) corresponding to 
# the direction vectors of each bval.
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
nvoxels = np.sum(mask) # nvoxels: no. of voxels used for calc. of response function
response, ratio = response_from_mask_ssst(gtab, data, mask)


# print("Shape of data is "+str(data.shape))
noise_list = ['gaussian','zero_noise']
percent_noise_list = [20,40,60,80] # list of percentage of noisy voxels
algo_list=['cholesky','qr','svd'] # list of algorithms
#Control variables
noise_type = None # Select out of None, 'gaussian', 'zero_noise'
fraction_noisy_voxels  = 0 # Wont run if noise_type is None
data_small = data[15:46, 55:87, 45:76]
# print("shape of data_small is:" + str(data_small.shape))
csd_fit = True # runs the CSD fitting procedure
read_data = False # to read files, code segment below
algo = algo_list[0]
num_csd_loops = 10
plot_data = False




if csd_fit:
	from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
	# for fraction_noisy_voxels in [0.2,0.4,0.6,0.8]:
		# for noise_type in noise_list:
	for algorithm in algo_list:
		csd_model = ConstrainedSphericalDeconvModel(gtab, response, noise_type=None,fraction_noisy_voxels=0,algo=algorithm)
		time_array = np.zeros(num_csd_loops)
		for i in range(num_csd_loops):
			start = time.time()
			csd_fit = csd_model.fit(data_small)
			end=time.time()
			time_array[i]=end-start
		# print(time_array)
		print("algorithm:%s, Mean Time for CSD(s):%f, Standard Deviation: %f" %(algorithm,np.mean(time_array),np.std(time_array)))

'''Following code fragment is for finding out the norm difference of fODF arrays produced
by different techniques. For eg: you could find out the fODF value by cholesky with no noise
and numpy least squares with 20 percent noise and find out the difference between them and plot
them to see the perturbation effect of induced noise.'''
if read_data:

	cholesky_fodf = pd.read_csv("./cholesky_None_0.csv").values
	for algorithm in algo_list:
		for noise_type in noise_list:
			for percent_noise in percent_noise_list:
				filename = str(algorithm)+"_"+str(noise_type)+"_"+str(percent_noise)+".csv"
				other_fodf= pd.read_csv(filename).values
				residual_norm = np.linalg.norm((cholesky_fodf-other_fodf),ord=2,axis=1)
				cholesky_norm = np.linalg.norm((cholesky_fodf),ord=2,axis=1)
				sum_percent=0.0
				count=0
				for i in range(residual_norm.shape[0]):
					if cholesky_norm[i]!=0 :
						sum_percent+=residual_norm[i]/cholesky_norm[i]*100
						count+=1
				print("algorithm:%s, percent_noise:%f ,noise_type:%s,norm difference percent:%f, count%f" %(algorithm, percent_noise,noise_type,(sum_percent/count),count))
				
				if plot_data:
					_ = plt.hist(norm_percent,bins='auto')
					plt.xlabel("differece(%)", fontsize = 12)
					plt.ylabel("No. of voxels", fontsize = 12)
					plt.title("Histogram", fontsize = 12)
					plotname = str(algorithm)+"_gaussian"+"_"+str(percent_noise)+".jpg"
					plt.savefig(plotname,dpi=300)
			
# A = np.random.randn(3,3)
# print(A)
# print(gaussian_noisifier(A,0.8))