import numpy as np
import time
from utils import soft_threshold, suo_init, suo_init_K

# I just coded this from scratch, hopefully it's efficient enough - perhaps a Cython implementation somewhere?
def bin_search(f,a0,b0,tol=10**-12):
    """Find zero of singlue argument function f given points a0 and b0 either side"""
    # Notice if a starting point is zero
    if f(a0)==0: return a0
    if f(b0)==0: return b0

    # Algo below written under first setting
    if f(b0) > 0 and f(a0) < 0:
        (a,b) = (a0,b0)
    # In other setting need to switch around start and end points
    elif f(a0) > 0 and f(b0) < 0:
        (a,b) = (b0,a0)
    else: raise ValueError('Start points must have different signs under f')

    while abs(b - a) > tol:
        c = 0.5*(a+b)
        if f(c)>0:
            b = c
        elif f(c) == 0:
            a = b = c
        else:
            a = c
    return a

# clue is in the name
def s_thresh_n_scale(w,delta):
    z = soft_threshold(w,delta)
    return z/np.linalg.norm(z,ord=2)

# most effect from soft thresholding when all but maximal elements sent zero
# i.e. delta between second biggest and biggest absolute values
def max_delta(w):
    try: top2 = np.unique(np.abs(w))[-2:]
    except IndexError:
        print('all elements of candidate for thresholding were the same, so no suitable delta')
        raise
    return np.mean(top2)

# Single step using binary search to find best delta (uses three functions above)
def PMD_update(w,c):
    w0 = s_thresh_n_scale(w,0)
    if np.linalg.norm(w0,ord=1) <= c: return w0
    else:
        delta_max = max_delta(w)
        f = lambda delta: np.linalg.norm(s_thresh_n_scale(w,delta),ord=1) - c
        delta_crit = bin_search(f,0,max_delta(w))
        return s_thresh_n_scale(w,delta_crit)


# Utility to determine convergence in `PMD`
def joint_in_tol(u1,u2,v1,v2,tol):
    return (np.linalg.norm(np.concatenate([u1-u2,v1-v2]))<tol)

# Find single canonical pair
def PMD(M,c1,c2,v0 = None, tol=10**-4):
    #initialise
    (m,p) = M.shape
    u0 = np.zeros(m)
    if v0 is None: v0 = np.ones(p)
    v0 = v0/np.linalg.norm(v0,ord=2)
    convergedq = False
    u,v = u0,v0

    #main loop
    while not convergedq:
        u_,v_ = u,v
        u = PMD_update(M @ v,c1)
        v = PMD_update(M.T @ u,c2)

        convergedq = joint_in_tol(u,u_,v,v_,tol)

    # final scaling
    d = u.T @ M @ v

    return (u,v,d)

# Find first K canonical pairs
# It would be good to have a check of validity of cs here
def PMD_CCA(X,Y,cs):
    start = time.time()
    K = len(cs)
    M = X.T @ Y
    (m,p) = M.shape
    Uest = np.zeros((m,0)); Vest = np.zeros((p,0))
    Dest = np.zeros(0)

    Uinit,Vinit = suo_init_K(X,Y,K=K)

    for k in range(K):
        c1 = c2 = cs[k]
        v0 = Vinit[:,k]
        # update objective matrix
        if k>0:
            M = M - Dest[k-1] * Uest[:,k-1].reshape(m,1)@Vest[:,k-1].reshape(p,1).T
        # run iteration
        u,v,d = PMD(M,c1,c2,v0=v0)
        Uest = np.block([Uest,u.reshape(m,1)])
        Vest = np.block([Vest,v.reshape(p,1)])
        Dest = np.block([Dest,d])

    te = time.time()-start
    return Uest,Vest,Dest,te

def PMD_first_cc(X,Y,c1,c2):
    start = time.time()
    u0,v0 = suo_init(X,Y)
    M = X.T @ Y
    u,v = PMD(M,c1,c2,v0=v0)[:2] #don't want to return d
    end = time.time()
    return  (u,v,end-start)
