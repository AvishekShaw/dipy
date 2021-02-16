import numpy as np
import random
from scipy.optimize import least_squares
from scipy.linalg import (svd,qr,solve)
import scipy.linalg.lapack as ll
import scipy.linalg as la
# from dipy.reconst.csdeconv import (_solve_cholesky)

def gaussian_noisifier_matrix(A,fraction_noisy_voxels):
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

def gaussian_noisifier_vector(s,fraction_noisy_voxels):
	'''
	Adds gaussian noise to a 1D vector
	Paramters:
	-----------
	s : 1D vector
		vector on which noise is to be added
	fraction_noisy_voxels : decimal value between 0 and 1
		The fraction of total voxels where noise it to be applied

	Returns:
	----------
	s : noisified 1D vector
	'''
	
	if s.ndim!=1 : raise ValueError("The input vector should be 1D")
	if (fraction_noisy_voxels>1 or fraction_noisy_voxels<0) :
		raise ValueError("The fraction of voxels to be noisified should between 0 and 1")

	m = s.shape[0]
	num_noisy_voxels = int(np.floor(m*fraction_noisy_voxels))
	random_ele = random.sample(range(0,m),num_noisy_voxels)
	for i in random_ele:
		s[i] += np.random.normal(loc=0,scale=np.std(s))
	return s

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


def solve_cholesky(Q, z):
	potrf, potrs = ll.get_lapack_funcs(('potrf', 'potrs'))
	L, info = potrf(Q, lower=False, overwrite_a=False, clean=False)
	if info > 0:
		msg = "%d-th leading minor not positive definite" % info
		raise la.LinAlgError(msg)
	if info < 0:
		msg = 'illegal value in %d-th argument of internal potrf' % -info
		raise ValueError(msg)
	f, info = potrs(L, z, lower=False, overwrite_b=False)
	if info != 0:
		msg = 'illegal value in %d-th argument of internal potrs' % -info
		raise ValueError(msg)
	return f

def solve_noisy_cholesky(A,b,fraction_noisy_voxels):
	U,s,Vh = svd(A)
	s = gaussian_noisifier_vector(s,fraction_noisy_voxels)
	A = U@np.diag(s)@Vh
	return solve_cholesky(A,b)

def solve_qr(A,b):
	Q,R = qr(A)
	y = Q.T@b
	m,n = R.shape
	x=np.zeros((m,1)) 
	x[m-1] = y[m-1]/R[m-1,n-1]
	for i in range(m-2,-1,-1):
		x[i] = (y[i] - np.dot(R[i,i+1:n].reshape(1,-1),x[i+1:m].reshape(-1,1)))/R[i,i]
	return x

def solve_noisy_qr(A,b,fraction_noisy_voxels):
	U,s,Vh = svd(A)
	s = gaussian_noisifier_vector(s,fraction_noisy_voxels)
	A = U@np.diag(s)@Vh
	return solve_qr(A,b)

def solve_svd(A,b):
    U,s,Vh = svd
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    x = np.dot(Vh.T,w)
    return x

def solve_truncated_svd(A,b,n_components):
    U,s,Vh = svd(A)
    U=U[:,0:n_components]
    s=s[0:n_components]
    Vh=Vh[0:n_components,:]
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    x = np.dot(Vh.T,w)
    return x

def solve_noisy_svd(A,b,fraction_noisy_voxels):
	"""
	Add noise to the singular values of A and solve Ax=b with SVD.
	"""
	U,s,Vh = svd(A)
	s = gaussian_noisifier_vector(s,fraction_noisy_voxels)
	c = np.dot(U.T,b)
	w = np.dot(np.diag(1/s),c)
	x = np.dot(Vh.T,w)
	return x

def solve_DSM(A,b,x_initial):
	m = x_initial.shape[0]
	del_x = np.zeros_like(x_initial)
	for _ in range(2):
		for i in range(m):
			print("Calculation for %d component starts",i)
			Numerator = Denominator = 0.0
			for j in range(m):
				print("value of j is" +str(j))
				print("b[j]=%f",b[j])
				print("Ax[j]=%f",(A@x_initial)[j])
				print("A[j,i]= %f",A[j,i])
				Numerator += (b[j] - (A@x_initial)[j])/A[j,i]
				print("Num+= %f", ((b[j] - (A@x_initial)[j])/A[j,i]))
				Denominator += (A@x_initial)[j]/A[j,i]
				print("Denom+= %f", ((A@x_initial)[j]/A[j,i]))
			del_x[i]+=Numerator/Denominator
		x_initial += del_x
		print("x value" + str(np.ndarray.flatten(x_initial)))
	return x_initial

a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype = np.float)
b= np.array([[7],[8],[9]],dtype = np.float)
x_initial = np.array([[1],[2],[3]],dtype = np.float)

# solve_DSM(a,b,x_initial)