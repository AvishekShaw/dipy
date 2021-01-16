import numpy as np
import random
from scipy.optimize import least_squares
from scipy.linalg import (svd,qr,solve)

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
	random_ele = random.sample(range(0,m*n),num_noisy_voxels) # to sample from the elements of the matrix
	for i in random_ele:
		A[i//n,i%n] += np.random.normal(loc=0,scale=np.std(A))
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
	random_ele = random.sample(range(0,m*n),num_noisy_voxels) # to sample from the elements of the matrix
	for i in random_ele:
		# if (i%(n+1)==0):pass
		A[i//n,i%n] = 0
	return(A)

def unconstrained_objective(x,A,b,H):
	"""
	Returns the objective function of minimization of Constrained Spherical Deconvolution.
	The equation is assumed to be ||Ax-b||^2 + ||Hx||^2.
	"""
	# print(A.shape,b.shape,H.shape)
	return(np.linalg.norm(A@x-b,ord=2,axis=1) +  np.linalg.norm(H@x,ord=2,axis=1))

def constrained_objective(x,A,b):
	"""Constrained Least squares calling function"""
	# print(A.shape,b.shape)
	return(A@x - b)
	# jac = A
	# x0= np.random.randn(45)
	# res = least_squares(fun,x0,jac=jac,bounds=(0,np.inf),args=(A,b),verbose=1)
	# print(res.x)

def solve_svd(A,b):
    # compute svd of A
    U,s,Vh = svd(A)

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T,b)
    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    w = np.dot(np.diag(1/s),c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = np.dot(Vh.conj().T,w)
    return x

def solve_qr(A,b):
	Q,R = qr(A)
	y = Q.T@b
	m,n = R.shape
	x=np.zeros((m,1))
	#back substitution 
	x[m-1] = y[m-1]/R[m-1,n-1]
	for i in range(m-2,-1,-1):
		x[i] = (y[i] - np.dot(R[i,i+1:n].reshape(1,-1),x[i+1:m].reshape(-1,1)))/R[i,i]
	return(x)



