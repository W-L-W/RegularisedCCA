import numpy as np
from typing import Union

from src.utils.linalg import nsqrtm, mhalf, overlap_to_sq_trigs, symmetric_pinv

def cca_from_cov_mat(Sig, p, zero_cut_off: Union[float, None] = None):
    """
    Returns (U,V,R) where the columns of U and V are the canonical directions and R is the vector of canonical correlations
    Uses np.linalg.inv and np.linalg.svd and np.linalg.eigh to implement via the SVD formulation of CCA
    Warning: convention that weights are the columns of U and COLUMNS of V is different to that of np.linalg.svd
    zero_cut_off some float turns columns of U,V corresponding to very small correlations to zero...
    """

    Sigxx, Sigxy, Sigyx, Sigyy = Sig[:p, :p], Sig[:p, p:], Sig[p:, :p], Sig[p:, p:]

    nSigxx, nSigyy = nsqrtm(Sigxx), nsqrtm(Sigyy)
    # Target_M = np.linalg.inv(sqrtm(Sigxx))@Sigxy@np.linalg.inv(sqrtm(Sigyy))
    Target_M = nSigxx @ Sigxy @ nSigyy

    # using notation from the paper, A, B are in normalised direction space
    A, R, B = np.linalg.svd(Target_M, full_matrices=False)
    # added full_matrices=False on 19 July after zero-cutoff gave me problems
    # Need to rescale by nuisance covariance to obtain ccs from SVD
    U = nSigxx @ A
    V = nSigyy @ B.T  # because np.linalg.svd returns V^T

    if zero_cut_off is not None:
        R_cut = (np.abs(R) > zero_cut_off)
        U = U @ np.diag(R_cut)
        V = V @ np.diag(R_cut)

    return (U, V, R)

def data_from_covariance(Sig,m,p,n,rs=0):
    """Generate n samples from X\in R^m, Y \in R^p and joint covariance Sig""" 
    # previously had some scaling by 1/sqrt(n), think stopped for CV integration
    rng = np.random.default_rng(seed=rs)
    data = rng.multivariate_normal(np.zeros(m+p),Sig,n)
    data -= data.mean(axis=0)
    X = data[:,:m]
    Y = data[:,m:]
    return (X,Y)



# EVALUATING ESTIMATES
######################

def emp_cor(u,v,X,Y):
    den = np.sqrt(u.T@X.T@X@u * v.T@Y.T@Y@v)
    if den == 0: return np.nan
    else: return (u.T@X.T@Y@v/den).item()

def true_cor(u,v,Sig):
    m = len(u); p = len(v)
    Sigxx,Sigxy,Sigyx,Sigyy = Sig[:m,:m],Sig[:m,m:],Sig[m:,:m],Sig[m:,m:]
    den = np.sqrt(u.T@Sigxx@u * v.T@Sigyy@v)
    if den == 0: return np.nan
    else: return (u.T@Sigxy@v/den).item()

def emp_cor_mat(Ue,Ve,X,Y):
    Uscaled = Ue @ np.diag(mhalf(np.diag(Ue.T@X.T@X@Ue)))
    Vscaled = Ve @ np.diag(mhalf(np.diag(Ve.T@Y.T@Y@Ve)))
    return Uscaled.T@X.T@Y@Vscaled

def true_cor_mat(Ue,Ve,Sig):
    m = Ue.shape[0]; p = Ve.shape[0]
    Sigxx,Sigxy,Sigyx,Sigyy = Sig[:m,:m],Sig[:m,m:],Sig[m:,:m],Sig[m:,m:]

    Uscaled = Ue @ np.diag(mhalf(np.diag(Ue.T@Sigxx@Ue)))
    Vscaled = Ve @ np.diag(mhalf(np.diag(Ve.T@Sigyy@Ve)))
    return Uscaled.T @ Sigxy @ Vscaled

def emp_cov_mat(Ue,Ve,X,Y):
    """We assume high dimensional case - so making X.T X is very expensive"""
    XUe = X @ Ue
    YVe = Y @ Ve
    Cxx = XUe.T @ XUe
    Cyy = YVe.T @ YVe
    Cxy = XUe.T @ YVe
    Cyx = Cxy.T
    return np.block([[Cxx,Cxy],[Cyx,Cyy]])

def true_cov_mat(Ue,Ve,Sig):
    p = Ue.shape[0]; q = Ve.shape[0]
    err_msg = f"Sig dims {Sig.shape} doesn't match U,V dims {p,q}"
    assert Sig.shape == (p+q, p+q), err_msg

    Sigxx,Sigxy,Sigyx,Sigyy = Sig[:p,:p],Sig[:p,p:],Sig[p:,:p],Sig[p:,p:]

    Cxx = Ue.T @ Sigxx @ Ue
    Cyy = Ve.T @ Sigyy @ Ve
    Cxy = Ue.T @ Sigxy @ Ve
    Cyx = Cxy.T

    return np.block([[Cxx,Cxy],[Cyx,Cyy]])


def can_corr_subs(cov_mat,K,agg='sq'):
    """Let C be cov mat of (X,Y) then return \sum agg(\rho_l) where (\rho_l)_l are ccs of X_{:k}, Y_{:k} for each k """
    assert cov_mat.shape == (2*K,2*K)
    def get_corr(k):
        ids = np.r_[:k,K:K+k]
        slice2d = np.ix_(ids,ids)
        C = cov_mat[slice2d]
        _,_,R = cca_from_cov_mat(C,k,zero_cut_off=None)
        if agg == 'sq': return np.sum(R**2)
        elif agg == 'sum': return np.sum(R)
    return np.array([get_corr(k) for k in range(1,K+1)])

def oracle_overlap(U,V,Sig):
    """Improved version; will copy over docstring later"""
    # assert that dimensions are consistent
    p,K = U.shape
    q,K2 = V.shape
    assert K==K2
    assert Sig.shape == (p+q,p+q)
    # separate into within and between view block covariances
    Sigxx,Sigxy,Sigyx,Sigyy = Sig[:p,:p],Sig[:p,p:],Sig[p:,:p],Sig[p:,p:]
    def upper_cholesky(A):
        """Return upper triangular matrix R such that A = R @ R.T
        To do so, reverse order of columns, compute cholesky, reverse back"""
        reversed_A = A[::-1,::-1]
        reversed_R = np.linalg.cholesky(reversed_A)
        return reversed_R[::-1,::-1] 
    
    # get this 'upper-cholesky' decomposition of within-view precision matrices
    Ru = upper_cholesky(symmetric_pinv(U.T @ Sigxx @ U))
    Rv = upper_cholesky(symmetric_pinv(V.T @ Sigyy @ V))

    return Ru.T @ (U.T @ Sigxy @ V) @ Rv

def oracle_overlap_old(U,V,Sig):
    """Return the target matrix for SVD for CCA with successive-subspace property
    Parameters:
    U: ndarray, shape (p,K)
    V: ndarray, shape (q,K)
    Sig: ndarray, shape (p+q,p+q)"""

    # assert that dimensions are consistent
    p,K = U.shape
    q,K2 = V.shape
    assert K==K2
    assert Sig.shape == (p+q,p+q)

    # separate into within and between view block covariances
    Sigxx,Sigxy,Sigyx,Sigyy = Sig[:p,:p],Sig[:p,p:],Sig[p:,:p],Sig[p:,p:]

    def upper_cholesky(A):
        """Return upper triangular matrix R such that A = R @ R.T
        To do so, reverse order of columns, compute cholesky, reverse back"""
        reversed_A = A[::-1,::-1]
        reversed_R = np.linalg.cholesky(reversed_A)
        return reversed_R[::-1,::-1]    
    try: 
        # get this 'upper-cholesky' decomposition of within-view precision matrices
        Ru = upper_cholesky(symmetric_pinv(U.T @ Sigxx @ U))
        Rv = upper_cholesky(symmetric_pinv(V.T @ Sigyy @ V))
        return Ru.T @ (U.T @ Sigxy @ V) @ Rv
    except:
        # save the offending matrices for debugging
        np.save('Mx.npy', U.T @ Sigxx @ U)
        np.save('My.npy', V.T @ Sigyy @ V)
        #print(np.linalg.pinv(V.T @ Sigyy @ V,hermitian=True).round(3))
        for M in (U.T @ Sigxx @ U, V.T @ Sigyy @ V):
            w, v = np.linalg.eigh(M)
            print('Min strictly less than zero?', min(w) < 0)
            print('Approximately symmetric?', np.allclose(M,M.T))
    # so Ru is upper triangular matrix such that Ru.T @ Ru = (U.T @ Sigxx @ U)^{-1}
    # so U @ Ru has Sigxx-orthonormal columns
    # moreover, the span of first k columns of U@Ru is the same as the span of first k columns of U
    # (we've effectively done Gram-Schmidt in the Sigxx inner product)

    # to get the target matrix quicker to evaluate multiplications as below (K << p,q)
    

def oracle_cos2thetas(U,V,Sig):
    """Compute successive subspace cos2thetas from given projection matrices"""
    # get target matrix
    target = oracle_overlap(U,V,Sig)
    # compute cos2thetas
    return overlap_to_sq_trigs(target,mode='cos')



# INITIALISATION FOR SCCA
#########################

def soft_threshold(M,gamma):
    ind1 = (M > + gamma)
    ind2 = (M < - gamma)
    return ind1*(M - gamma) + ind2*(M + gamma)

def suo_init(X,Y,gamma=None):
    if gamma:
        Sxy = soft_threshold(X.T@Y,gamma)
    else:
        thresh = np.diag(X.T@X).reshape(-1,1)@np.diag(Y.T@Y).reshape(1,-1)
        n = X.shape[0]
        thresh = 2.5*np.sqrt(thresh/n)
        Sxy = soft_threshold(X.T@Y,thresh)

    Uh,D,Vh = np.linalg.svd(Sxy)
    Vh = Vh.T
    #np.isclose(Sxy,Uh@np.diag(D)@Vh[:m,:])

    Uscaling = np.sqrt(np.diag(Uh.T@X.T@X@Uh))
    Vscaling = np.sqrt(np.diag(Vh.T@Y.T@Y@Vh))

    Util = Uh/Uscaling
    Vtil = Vh/Vscaling

    Dtil = Util.T@X.T@Y@Vtil
    k = np.argmax(np.diag(Dtil))
    u_init = Util[:,k]
    v_init = Vtil[:,k]

    return (u_init,v_init)

def suo_init_K(X,Y,K,gamma=None):
    if gamma:
        Sxy = soft_threshold(X.T@Y,gamma)
    else:
        thresh = np.diag(X.T@X).reshape(-1,1)@np.diag(Y.T@Y).reshape(1,-1)
        n = X.shape[0]
        thresh = 2.5*np.sqrt(thresh/n)
        Sxy = soft_threshold(X.T@Y,thresh)

    Uh,D,Vh = np.linalg.svd(Sxy)
    Vh = Vh.T
    #np.isclose(Sxy,Uh@np.diag(D)@Vh[:m,:])

    Uscaling = np.sqrt(np.diag(Uh.T@X.T@X@Uh))
    Vscaling = np.sqrt(np.diag(Vh.T@Y.T@Y@Vh))

    Util = Uh/Uscaling
    Vtil = Vh/Vscaling

    Dtil = Util.T@X.T@Y@Vtil
    ks = np.diag(Dtil).argsort()[-K:][::-1]
    return Util[:,ks], Vtil[:,ks]
