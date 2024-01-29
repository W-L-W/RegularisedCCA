import numpy as np

from scipy.linalg import toeplitz
from scipy.stats import ortho_group

from typing import List, Union
# some custom types, for readability
Vector = np.ndarray
Array2D = np.ndarray
PSDMatrix = np.ndarray

# CANONICAL PAIR MODEL CONSTRUCTION
###################################
def joint_cov_from_within_view_covs_and_candidate_directions(
        Sigxx, Sigyy, U_cand, V_cand, R, 
    ):
    # check dimensions are consistent
    p = Sigxx.shape[0]; q = Sigyy.shape[0]; K = R.shape[0]
    assert Sigxx.shape == (p,p) and Sigyy.shape == (q,q)
    assert U_cand.shape == (p,K) and V_cand.shape == (q,K)

    U = gram_schmidt(U,Sigxx)
    V = gram_schmidt(V,Sigyy)    

    Sigxy = Sigxx @ U @ R @ V.T @ Sigyy
    Sigyx = Sigxy.transpose()
    Sig = np.block([[Sigxx,Sigxy],[Sigyx,Sigyy]])
    return Sig

# change implementation of GS to do normalisation implicitly TODO
def make_o_n(orig: Vector, o_n_list: List[Vector], psd: PSDMatrix):
    """Given list of o.n. vectors, a psd matrix, and a vector orig, return a vector orthogonal to all o.n. vectors w.r.t. psd inner product"""
    if o_n_list is []: return orig
    else: 
        orig_proj = orig - sum([perp * (orig.T @ psd @ perp).item() for perp in o_n_list])
        norm_2 = (orig_proj.T @ psd @ orig_proj).item()
        return orig_proj /  np.sqrt(norm_2)

def gram_schmidt(input: Union[List[Vector], Array2D],psd_mat: PSDMatrix) -> Union[List[Vector], Array2D]:
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
    


# CONSTRUCTING WITHIN-VIEW COVARIANCE MATRICES
##############################################
# An implementation of 'Identity-like covariance models' from Suo paper
def iden_like_cov_sub(method,k):
    if method == 'identity':
        generating_pattern = [1] + [0]*(k-1)
        return toeplitz(generating_pattern)
    elif method == 'toeplitz':
        generating_pattern = 0.9**np.arange(k)
        return toeplitz(generating_pattern)
    elif method == 'sparse':
        generating_pattern = [1,0.5,0.4] + [0]*(k-3)
        return np.linalg.inv(toeplitz(generating_pattern))
    elif method == 'low_rank':
        # we want reproducibility so fix random state
        M = ortho_group.rvs(dim=k, random_state=0)
        pdef = M @ np.diag([0.9**j+0.2 for j in range(k)]) @ M.T
        return pdef
    else:
        raise NameError('method needs to be identity, toeplitz or sparse')




# TOY CONSTRUCTIONS FOR JOINT COVARIANCE MATRICES
#################################################
def suo_iden_like_cov(method, u, v, rho):
    # note, could implement with joint_cov_from_within_view_covs_and_candidate_directions
    # but left with explicit construction here for clarity
    p = len(u)
    q = len(v)

    Sigxx = iden_like_cov_sub(method, p)
    Sigyy = iden_like_cov_sub(method, q)

    u = u.reshape(p, 1)
    v = v.reshape(q, 1)
    u = u / np.sqrt(u.T @ Sigxx @ u)
    v = v / np.sqrt(v.T @ Sigyy @ v)

    Sigxy = rho * Sigxx @ u @ v.T @ Sigyy
    Sigyx = Sigxy.transpose()

    Sig = np.block([[Sigxx, Sigxy], [Sigyx, Sigyy]])
    return Sig

def suo_basic(method,m,p,rho=0.9,ms=5,ps=5):
    u = np.concatenate([np.ones(ms),np.zeros(m-ms)])
    v = np.concatenate([np.ones(ps),np.zeros(p-ps)])
    Sig = suo_iden_like_cov(method,u,v,rho=rho)
    return Sig

def suo_rand(method,m,p,rho=0.9,ms=5,ps=5):
    rng = np.random.default_rng(seed=0)
    u = np.concatenate([rng.uniform(low=-1,high=1,size=ms),np.zeros(m-ms)])
    v = np.concatenate([rng.uniform(low=-1,high=1,size=ps),np.zeros(p-ps)])
    Sig = suo_iden_like_cov(method,u,v,rho=rho)
    return Sig

# an example of a more complicated structure using the canonical pair model
def geom_corr_decay_sparse_weight_multi_spike(
        p, q, K, decay_ratio, spike_size, method='identity', geom_param=1
    ):
    rng = np.random.default_rng(seed=0)
    U,V = np.zeros((p,K)),np.zeros((q,K))
    Sigxx = iden_like_cov_sub(method,p)
    Sigyy = iden_like_cov_sub(method,q)

    def get_probs(d):
        probs = np.array([geom_param**k for k in range(d)])
        return probs / sum(probs)
    probsp = get_probs(p); probsq = get_probs(q)

    for k in range(K):
        u_ids = rng.choice(p,size=spike_size,replace=False,p=probsp)
        v_ids = rng.choice(q,size=spike_size,replace=False,p=probsq)
        U[u_ids,k] = rng.uniform(low=-1,high=1,size=spike_size)
        V[v_ids,k] = rng.uniform(low=-1,high=1,size=spike_size)
    
    R = np.diag([decay_ratio**k for k in range(1,K+1)])

    return joint_cov_from_within_view_covs_and_candidate_directions(
        Sigxx, Sigyy, U, V, R
    )

# inspired by Bach and Jordan Probabilistic CCA 2005
def cov_from_latents(p,q,nlatents,decay_ratio,supp_size,method='identity',geom_param=1):
    rng = np.random.default_rng(seed=0)
    Wx,Wy = np.zeros((p,nlatents)),np.zeros((q,nlatents))
    Psix = iden_like_cov_sub(method,p)
    Psiy = iden_like_cov_sub(method,q)

    def get_probs(d):
        probs = np.array([geom_param**k for k in range(d)])
        return probs / sum(probs)
    probsp = get_probs(p); probsq = get_probs(q)

    for k in range(nlatents):
        x_ids = rng.choice(p,size=supp_size,replace=False,p=probsp)
        y_ids = rng.choice(q,size=supp_size,replace=False,p=probsq)
        Wx[x_ids,k] = rng.uniform(low=-1,high=1,size=supp_size)
        Wy[y_ids,k] = rng.uniform(low=-1,high=1,size=supp_size)

    D = 2*np.abs(np.diag([(nlatents - 2*k)/nlatents * decay_ratio**k for k in range(1,nlatents+1)]))
    Wx = Wx @ D
    Wy = Wy @ D

    Sigxx = Psix + Wx @ Wx.T
    Sigyy = Psiy + Wy @ Wy.T
    Sigxy = Wx @ Wy.T
    Sigyx = Sigxy.T

    Sigma = np.block([[Sigxx,Sigxy],[Sigyx,Sigyy]])
    return Sigma