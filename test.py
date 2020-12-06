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








hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

# Data has a shape of (81,106,76,160). 160 is the number of directions
data, affine = load_nifti(hardi_fname)
#print("The shape of data is:" + str(data.shape))

noise_type = 'gaussian'
fraction_noise  = .2
data_small = data[20:51, 55:86, 38:69]
save_fig = False
csd_fit = False


# There are 160 bvals corresponding to 160 directions
# and bvecs is an array of (160,3) corresponding to 
# the direction vectors of each bval.

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
# print(bvals.shape,bvecs.shape)

gtab = gradient_table(bvals, bvecs)

from dipy.reconst.csdeconv import (auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.0)

# nvoxels is the number of voxels used for calculation of response function
nvoxels = np.sum(mask)


response, ratio = response_from_mask_ssst(gtab, data, mask)


if save_fig:

	from dipy.viz import window, actor
	from dipy.sims.voxel import single_tensor_odf

	# Enables/disables interactive visualization
	interactive = False

	scene = window.Scene()
	evals = response[0]
	evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T


	response_odf = single_tensor_odf(default_sphere.vertices, evals, evecs)
	# transform our data from 1D to 4D
	response_odf = response_odf[None, None, None, :]
	response_actor = actor.odf_slicer(response_odf, sphere=default_sphere,
	                                  colormap='plasma')
	scene.add(response_actor)
	print('Saving illustration as csd_response.png')
	window.record(scene, out_path='csd_response.png', size=(200, 200))
	if interactive:
	    window.show(scene)




from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
csd_model = ConstrainedSphericalDeconvModel(gtab, response, noise_type,fraction_noise)


if csd_fit:

	for _ in range(1):
		start = time.time()
		csd_fit = csd_model.fit(data_small)
		end=time.time()
		# print(end-start, "\n")





if save_fig:

	csd_odf = csd_fit.odf(default_sphere)

	fodf_spheres = actor.odf_slicer(csd_odf, sphere=default_sphere, scale=0.9,
	                                norm=False, colormap='plasma')

	scene.add(fodf_spheres)

	print('Saving illustration as csd_odfs.png')
	window.record(scene, out_path='csd_odfs.png', size=(600, 600))
	if interactive:
	    window.show(scene)

	from dipy.direction import peaks_from_model

	csd_peaks = peaks_from_model(model=csd_model,
	                             data=data_small,
	                             sphere=default_sphere,
	                             relative_peak_threshold=.5,
	                             min_separation_angle=25,
	                             parallel=True)

	scene.clear()
	fodf_peaks = actor.peak_slicer(csd_peaks.peak_dirs, csd_peaks.peak_values)
	scene.add(fodf_peaks)

	print('Saving illustration as csd_peaks.png')
	window.record(scene, out_path='csd_peaks.png', size=(600, 600))
	if interactive:
	    window.show(scene)


read_data = True


'''Following code fragment is for finding out the norm difference of fODF arrays produced
by different techniques. For eg: you could find out the fODF value by cholesky with no noise
and numpy least squares with 20 percent noise and find out the difference between them and plot
them to see the perturbation effect of induced noise.'''

if read_data:
	cholesky = pd.read_csv("./cholesky_no_noise.csv").values
	lstsq = pd.read_csv("./lstsq_20_noise.csv").values
	print(lstsq.shape)

	residual_norm = np.linalg.norm((cholesky-lstsq),ord=2,axis=1)
	cholesky_norm = np.linalg.norm((cholesky),ord=2,axis=1)

	norm_percent = (residual_norm/cholesky_norm)*100
	_ = plt.hist(norm_percent,bins='auto')
	plt.xlabel("\%differece", fontsize = 12)
	plt.ylabel("No. of voxels", fontsize = 12)
	plt.title("Histogram comparing cholesky and np.lstsq with 20% noise", fontsize = 12)
	plt.savefig("hist.jpg",dpi=300)

	print(type(norm_percent))





