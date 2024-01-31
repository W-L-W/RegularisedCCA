import itertools
import numpy as np
import time

from typing import Union, List

Vector = np.ndarray
Matrix = np.ndarray
SqMatrix = np.ndarray
SymMatrix = np.ndarray
PSDMatrix = np.ndarray

# ROOTS, INVERSES, POSITIVE DEFINITENESS
########################################

def _masked_power(vec: Vector, power: float, atol: float =10**-6, rgln: float = 0):
    """Raise elements of a non-negative vector to the power of `power`, replacing zero entries with `rgln`"""
    w = vec.astype('float')
    mask = ~np.isclose(w, 0, atol=atol)
    w[mask] = (w[mask])**power
    w[~mask] = rgln
    return w

def mhalf(vec: Vector):
    """Raise elements of a non-negative vector to the power of (-1/2), leaving zero entries as zero"""
    # this masking implementation was significantly faster than using np.vectorize on large vectors
    assert (vec >= 0).all(), 'Input to square root must be non-negative'
    return _masked_power(vec, -0.5)

def sqrtm(M: PSDMatrix):
    w, v = np.linalg.eigh(M)
    return v@ np.diag(w**0.5) @v.T

def nsqrtm(M: PSDMatrix):
    w, v = np.linalg.eigh(M)
    w_nsqrt = mhalf(w)
    return v@ np.diag(w_nsqrt) @v.T

def symmetric_pinv(M: SymMatrix, rgln: float = 10**-8):
    """regularised pseudo-inverse with explicit lower bound of rgln on eigenvalues
    (np.linalg.pinv gave me non-positive inverses)"""
    w, v = np.linalg.eigh(M)
    w_inv = _masked_power(w, power=-1, rgln=rgln)
    return v@ np.diag(w) @v.T

def isPSD(B: Matrix):
    la = np.linalg
    #https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """Returns true when input is positive-definite, via Cholesky"""
    ## will use to check if estimated covariance matrices from gglasso are genuinely positive semi-definite 
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def get_near_psd(A: Matrix,regn=10**-8):
    #https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0 
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T) + regn * np.identity(C.shape[0])



# PROJECTIONS AND CANONICAL ANGLES
##################################

def proj_onto_col_space(A: Matrix):
    Q,R = np.linalg.qr(A)
    return Q@Q.T

def sin2theta(A: Matrix,B: Matrix):
    if len(A.shape)==1:
        A = A.reshape(-1,1)
    if len(B.shape)==1:
        B = B.reshape(-1,1)
    assert len(A.shape)==len(B.shape)==2
    assert A.shape[0] == B.shape[0]
    n,k = A.shape

    A_GS,_ = np.linalg.qr(A)
    B_GS,_ = np.linalg.qr(B)

    return k - np.linalg.norm(A_GS.T @ B_GS,ord='fro')**2

def sin2theta_mult(A: Matrix, B: Matrix):
    """Return 1D array of successive sin2theta values between successive column-spaces of A,B"""
    return sq_trigs(A,B,mode='sin')

def sq_trigs(A: Matrix, B: Matrix, mode='cos'):
    """Compute successive cos/sin2theta values between successive column-spaces of two matrices
    
    Parameters
    ----------
    A : ndarray, shape (p,K)
        First matrix of K columns
    B : ndarray, shape (p,K)
        Second matrix of K columns
    mode : {'cos','sin'}, optional
        Whether to return cos2 theta or sin2 theta, by default 'cos'
    """
    if len(A.shape)==1:
        A = A.reshape(-1,1)
    if len(B.shape)==1:
        B = B.reshape(-1,1)
    assert len(A.shape)==len(B.shape)==2
    assert A.shape[0] == B.shape[0]
    n,K = A.shape

    A_GS,_ = np.linalg.qr(A)
    B_GS,_ = np.linalg.qr(B)

    # take element-wise square of matrix of inner products
    return overlap_to_sq_trigs(A_GS.T @ B_GS,mode=mode)
    
def overlap_to_sq_trigs(overlap: SqMatrix, mode='cos'):
    """Converts overlap matrix to squared cosines or sines of canonical angles
    Parameters:
    overlap: ndarray, shape (K,K)
        (might want to make non-square later but should be fine for now)
    mode: {'cos','sin'}
    """
    sq_overlap = overlap**2
    K = overlap.shape[0]

    # cos2theta similarity for top-k space is sum of elements in top-k submatrix
    # can compute for all k at once by cumulatively summing entries in final row/columns of submatrices
    def added_signal(k):
        # return sum of elements in final row/column of top-k submatrix
        return np.sum(sq_overlap[k,:k]) + np.sum(sq_overlap[:k+1,k])

    signals = np.array([added_signal(k) for k in range(K)])
    cos2thetas = signals.cumsum()
    if mode=='cos':
        return cos2thetas
    
    elif mode=='sin':
        # sin2theta distance is 1 - cos2theta similarity, vectorised version
        sin2thetas = np.arange(1,K+1) - cos2thetas
        return sin2thetas



# REGISTRATION
##############
    
def register_general(Z1,Z0,reg_method='orthog',output='transformer_map',via=None):
    """Transforms big Z1 onto a smaller Z0 via partial isometry. i.e. dimensions below need k1>=k0
    
    Parameters
    ----------
    Z1 : np.array of shape (n,k1)
        The matrix to be registered

    Z0 : np.array of shape (n,k0)
        The reference matrix

    reg_method : str, optional
        The registration method. The default is 'orthog'. Other options are 'signs', 'perm_n_signs'. 
        (may implement more later too)

    output : str, optional
        The output type. The default is 'transformer_map'. Other options are 'transformed_Z'
    
    via : None or np.array of shape (n,n2), optional
        If provided, then the registration is done via this matrix.
        This is so we can apply this same function to U1, U0 via X meaning registering X@U1 to X@U0
        (Not sure if this is best coding practice, but it saves repeating code at least!)
    """
    if via is not None:
        Z1 = via @ Z1
        Z0 = via @ Z0
        return register_general(Z1,Z0,reg_method=reg_method,output=output,via=None)

    if reg_method=='signs':
        # this only makes sense if have same dimensions to start with, let's assert this
        assert Z1.shape[1]==Z0.shape[1], 'Z1 and Z0 must have same number of columns'

        signs = np.sign(np.sum(Z1*Z0, axis=0))
        signs[signs==0]=1
        if output=='transformer_map':
            return lambda M: M * signs
        elif output=='transformed_Z':
            return Z1 * signs
        
    elif reg_method=='perm_n_signs':
        k1 = Z1.shape[1]
        k0 = Z0.shape[1]
        # instead of permutations we only want lists of length k0 of elements from range(k1)
        perms_to_try = list(itertools.product(range(k1),repeat=k0))
        def score_p(p):
            return np.abs((Z0 * Z1[:,p]).sum(axis=0)).sum()
        score_dict = {p:score_p(p) for p in perms_to_try}
        best_perm = max(score_dict,key=score_dict.get)

        signs=np.sign(np.sum(Z1[:,best_perm]*Z0,axis=0))
        signs[signs==0]=1

        if output=='transformer_map':
            return lambda M: (M)[:,best_perm] * signs
        elif output=='transformed_Z':
            return (Z1)[:,best_perm] * signs
    
    elif reg_method=='orthog':
        u,d,vh = np.linalg.svd(Z1.T @ Z0,full_matrices=False)
        if output=='transformer_map':
            return lambda M: M @ u @ vh
        elif output=='transformed_Z':
            return Z1 @ u @ vh

    else:
        raise ValueError('reg_method must be one of "signs", "perm_n_signs", "orthog"') 
    

# CUSTOM GRAM-SCHMIDT
#####################
def make_o_n(orig: Vector, o_n_list: List[Vector], psd: PSDMatrix):
    """Given list of o.n. vectors, a psd matrix, and a vector orig, return a vector orthogonal to all o.n. vectors w.r.t. psd inner product"""
    if o_n_list is []: return orig
    else: 
        orig_proj = orig - sum([perp * (orig.T @ psd @ perp).item() for perp in o_n_list])
        norm_2 = (orig_proj.T @ psd @ orig_proj).item()
        return orig_proj /  np.sqrt(norm_2)

def gram_schmidt(input: Union[List[Vector], Matrix], psd_mat: PSDMatrix) -> Union[List[Vector], Matrix]:
    """Make vectors o.n. with respect to psd_mat-inner-product.
    Given: set of vectors (either in a list or successive columns of matrix), 
    Return: set of vectors who are psd_mat o.n. and whose successive spans are the same as the original set of vectors"""
    if type(input) == list:
        vec_list = input
        m,R = len(vec_list[0]),len(vec_list)
        new_list = []
        for r in range(R):
            new_list.append(make_o_n(vec_list[r],new_list,psd_mat))
        return new_list

    # to handle matrices whose successive columns are to be made o.n., we call the list implementation
    elif type(input) == np.ndarray:
        vec_list = list(input.T)
        new_list = gram_schmidt(vec_list,psd_mat)
        return np.array(new_list).T