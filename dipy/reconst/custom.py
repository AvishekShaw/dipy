import numpy as np
import random

def gaussian_noisifier(A,fraction_noisy_voxels):
	"""
	Function to add gaussian noise to a fraction of voxels of the matrix
	Paramters:
	-----------
	A : 2D matrix
		Matrix on noise is to be added
	fraction_noisy_voxels : decimal value between 0 and 1
		The fraction of total voxels where noise it to be applied

	Retunrs:
	----------
	A : noisified 2D matrix
	"""
	if A.ndim!=2 : raise ValueError("The input matrix should be 2D")
	if (fraction_noisy_voxels>1 or fraction_noisy_voxels<0) :
		raise ValueError("The fraction of voxels to be noisified should between 0 and 1")

	m,n = A.shape
	num_noisy_voxels = int(np.floor(m*n*fraction_noisy_voxels))
	random_list = random.sample(range(0,m*n),num_noisy_voxels)
	for i in random_list:
		A[i//n,i%n] += np.random.randn()
	return(A)


def add_zero_noise(A,fraction_noisy_voxels):
	"""
	Function to add zero noise to a fraction of voxels of the matrix where random
	elements of the matrix are made zero.
	Paramters:
	-----------
	A : 2D matrix
		Matrix on noise is to be added
	fraction_noisy_voxels : decimal value between 0 and 1
		The fraction of total voxels where noise it to be applied

	Retunrs:
	----------
	A : noisified 2D matrix
	"""
	if A.ndim!=2 : raise ValueError("The input matrix should be 2D")
	if (fraction_noisy_voxels>1 or fraction_noisy_voxels<0) :
		raise ValueError("The fraction of voxels to be noisified should between 0 and 1")
	
	m,n = A.shape
	num_noisy_voxels = int(np.floor(m*n*fraction_noisy_voxels))
	random_list = random.sample(range(0,m*n),num_noisy_voxels)
	for i in random_list:
		# if (i%(n+1)==0):pass
		A[i//n,i%n] = 0
	return(A)

