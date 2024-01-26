import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import sqrtm
from sklearn.preprocessing import normalize
import time

import utils

# initialisation, something I made up myself, previously in file 'suo_algo'
def get_initialisation(X,Y,r=0):
    """Given data, provides OK guess for the rth canonical correlation vectors"""
    Sx_hat = X.transpose()@X
    Sy_hat = Y.transpose()@Y

    #let's extract diagonal elements the invert this to help with stability
    #mrt for minus square root
    Sx_mrt = np.diag(np.diag(Sx_hat)**-0.5)
    Sy_mrt = np.diag(np.diag(Sy_hat)**-0.5)

    #Finally compute estimator of matrix to SVD:
    M_hat = Sx_mrt @ X.transpose() @ Y @ Sy_mrt

    #Get initalisation from SVD
    U,D,V = np.linalg.svd(M_hat)
    ur = U[:,r]
    vr = V[r,:]

    return (ur,vr)

# # method below is slightly more principled, especially for sparse case
# # however, implementing will require time to edit to later pairs - will leave for now
# def suo_init(X,Y,gamma=None):
#     """
#     Method of initialisation proposed in Suo paper.
#     Only defined there for first pair but easy to extend to first-r pairs.
#     """
#     if gamma:
#         Sxy = soft_threshold(X.T@Y,gamma)
#     else:
#         thresh = np.diag(X.T@X).reshape(-1,1)@np.diag(Y.T@Y).reshape(1,-1)
#         n = X.shape[0]
#         thresh = 2.5*np.sqrt(thresh/n)
#         Sxy = soft_threshold(X.T@Y,thresh)
#
#     Uh,D,Vh = np.linalg.svd(Sxy)
#     Vh = Vh.T
#     #np.isclose(Sxy,Uh@np.diag(D)@Vh[:m,:])
#
#     Uscaling = np.sqrt(np.diag(Uh.T@X.T@X@Uh))
#     Vscaling = np.sqrt(np.diag(Vh.T@Y.T@Y@Vh))
#
#     Util = Uh/Uscaling
#     Vtil = Vh/Vscaling
#
#     Dtil = Util.T@X.T@Y@Vtil
#     k = np.argmax(np.diag(Dtil))
#     u_init = Util[:,k]
#     v_init = Vtil[:,k]
#
#     return (u_init,v_init)



#Prerequisite functions and then main code for single ADMM step
def bigproxmf(x,mu,c,tau):
    ind1 = (x+mu*c > mu*tau)
    ind2 = (x+mu*c < -mu*tau)
    return ind1*(x+mu*c - mu*tau) + ind2*(x+mu*c + mu*tau)

def proxlg(x):
    mag = np.linalg.norm(x,ord=2)
    if (mag>1): return x/mag
    else: return x


def lADMM_opt(b,C,C_,D,a0,z0=None,xi0=None,tau=0.001,lam=10,eps_rel=10**-5,eps_abs=10**-5,tracked=False):
    """
    Use ADMM for a single optimisation of the biconvex objective for finding (r+1)th canonical vector
    We identify (a,b,C,D) with the (u,v,X,Y) from algorithm in appendix of Suo's paper
    (So treat b as fixed and find best a)
    In this formulation, the duals are most important - so will explicitly make them main arguments
    """
    #we have an auxilliary matrix I_top which helps us encode the orthogonality constraints
    n = C.shape[0]
    r = C_.shape[0] - n
    I_top = np.block([[np.identity(n)],[np.zeros([r,n])]])

    #define tolerances - these update as our estimates update (estimates can change a lot)
    p = C.shape[1]
    def get_eps():
        eps_pri = np.sqrt(p) * eps_abs + eps_rel * max(np.linalg.norm(C@ak),np.linalg.norm(zk))
        eps_dua = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(C_.transpose()*xik)
        return (eps_pri,eps_dua)

    #define convergence criterion
    def convq(sres,rres,eps_pri,eps_dua):
        return (rres <= eps_pri) & (sres <= eps_dua)

    #set the variable proximal operator
    c = C.transpose()@D@b
    mu=lam/(4*np.linalg.norm(C_,ord='fro'))
    proxmf = (lambda x: bigproxmf(x,mu,c,tau))

    #initialise; something sensible if have no duals to start with
    ak  = a0
    if (z0 is None) or (xi0 is None):
        print('resetting')
        zk  = proxlg(C@ak)
        xik = C_@ak-I_top@zk
    else:
        print('using inputs')
        zk = z0
        xik = xi0


    convergedq = False

    #tracked is a tool for debugging - just puts relevant step data into a list
    if tracked: track = []

    #main ADMM loop
    while not convergedq:
        #store old estimate
        ak_ = ak

        #update our estimates
        ak  = proxmf(ak - mu/lam*C_.transpose()@(C_@ak-I_top@zk+xik))
        zk  = proxlg(C@ak + I_top.T@xik)
        xik = xik + C_@ak - I_top@zk

        #now update our convergence tolerances
        (eps_pri,eps_dua) = get_eps()

        #compute residuals, and their norms - careful was previously typo in sk
        sk = (ak-ak_)/mu
        rk = C_@ak - I_top@zk
        sres = np.linalg.norm(sk)
        rres = np.linalg.norm(rk)

        #check if convergence criterion met
        convergedq = convq(sres,rres,eps_pri,eps_dua)
        scaled_rres = rres/eps_pri
        scaled_sres = sres/eps_dua

        #save estimates
        if tracked: track.append(
        {'sres':scaled_sres,'rres':scaled_rres,'ak':ak}
        )

    if tracked: return track
    else: return (ak,zk,xik)




#Prerequisite function then main code for finding arbitrary next canonical correlation
def joint_in_tol(u1,u2,v1,v2,tol):
    return (np.linalg.norm(np.concatenate([u1-u2,v1-v2]))<tol)

def next_ccs(u_init,v_init,X,Y,X_,Y_,tol=10**-4,tol_ADMM=10**-6, lam=10,tau=0.01,tracked=False):
    convergedq=False
    if tracked: track=[]
    uk,vk = u_init,v_init
    zuk = zvk = xiuk = xivk = None

    t = time.time()
    while not convergedq:
        uk_,vk_ = uk,vk
        t_ = t

        uk,zuk,xiuk = lADMM_opt(vk,X,X_,Y,a0=uk,z0=zuk,xi0=xiuk,tau=tau,lam=lam,eps_rel=tol_ADMM,eps_abs=tol_ADMM)
        vk,zvk,xivk = lADMM_opt(uk,Y,Y_,X,a0=vk,z0=zvk,xi0=xivk,tau=tau,lam=lam,eps_rel=tol_ADMM,eps_abs=tol_ADMM)

        convergedq = joint_in_tol(uk_,uk,vk_,vk,tol)

        t = time.time()
        if tracked:
            track.append(
            {'t':t-t_,'uk':uk_,'zuk':zuk,'xiuk':xiuk,'vk':vk_,'zvk':zvk,'xivk':xivk}
            )

    if tracked: return track
    else: return [uk,vk]


def first_r_ccs(X,Y,r_max,tol=10**-4,tau=0.01):
    m,p = X.shape[1],Y.shape[1]
    Uest = np.zeros((m,0))
    Vest = np.zeros((p,0))

    start = time.time()

    for r in range(r_max):
        print(f"Getting {r}th ccs")
        X_ = np.block([[X],[Uest.T @ X.T @ X]])
        Y_ = np.block([[Y],[Vest.T @ Y.T @ Y]])

        u_init,v_init = get_initialisation(X,Y,r=r)
        uf,vf = next_ccs(u_init,v_init,X,Y,X_,Y_,tol=tol,tau=tau)

        Uest = np.block([Uest,uf.reshape(-1,1)])
        Vest = np.block([Vest,vf.reshape(-1,1)])

    end = time.time()
    elapsed = end-start
    print ("Time elapsed:", elapsed)

    return Uest,Vest,elapsed


from utils import suo_init


def suo_first_cc(X,Y,tol=10**-4,tau=0.01):
    start = time.time()

    u_init,v_init = suo_init(X,Y)
    uf,vf = next_ccs(u_init,v_init,X,Y,X,Y,tol=tol,tau=tau)

    end = time.time()
    elapsed = end-start
    print ("Time elapsed:", elapsed)

    return uf.reshape(-1,1),vf.reshape(-1,1),elapsed



## Fixed number of ADMM steps...
def lADMM_steps(b,C,C_,D,a0,z0=None,xi0=None,tau=0.001,lam=10,n_steps=20,tracked=False):
    """
    Use ADMM for a single optimisation of the biconvex objective for finding (r+1)th canonical vector
    We identify (a,b,C,D) with the (u,v,X,Y) from algorithm in appendix of Suo's paper
    (So treat b as fixed and find best a)
    In this formulation, the duals are most important - so will explicitly make them main arguments
    """
    #we have an auxilliary matrix I_top which helps us encode the orthogonality constraints
    n = C.shape[0]
    r = C_.shape[0] - n
    I_top = np.block([[np.identity(n)],[np.zeros([r,n])]])

    #define tolerances - these update as our estimates update (estimates can change a lot)
    p = C.shape[1]

    #set the variable proximal operator
    #try:
    c = C.transpose()@D@b
        #print(f'C={C}')
    #except FloatingPointError as e:
    #    print('Houston, we have a warning:', e)
    #    #return b
    #    pass

    # I think it's an operator norm - which in numpy is ord='2'
    mu=lam/(2*np.linalg.norm(C_,ord=2)**2)
    proxmf = (lambda x: bigproxmf(x,mu,c,tau))

    #initialise; something sensible if have no duals to start with
    ak  = a0
    if (z0 is None) or (xi0 is None):
        #print('resetting') for debugging
        zk  = proxlg(C@ak)
        xik = C_@ak-I_top@zk
    else:
        #print('using inputs') #for debugging
        zk = z0
        xik = xi0

    #tracked is a tool for debugging - just puts relevant step data into a list
    if tracked: track = []

    #main ADMM loop
    for step in range(n_steps):
        #store old estimate
        ak_ = ak

        #update our estimates
        ak  = proxmf(ak - mu/lam*C_.transpose()@(C_@ak-I_top@zk+xik))
        zk  = proxlg(C@ak + I_top.T@xik)
        xik = xik + C_@ak - I_top@zk

        #save estimates
        if tracked: track.append(
        {'ak':ak}
        )

    if tracked: return track
    else: return (ak,zk,xik)


def next_ccs_lazy(u_init,v_init,X,Y,X_,Y_,n_steps_ACS=10,n_steps_ADMM=100, lam=3,tau=0.01,tracked=False):
    """
    Say this is lazy because we only go for a fixed number of ACS/ADMM steps rather than to convergence
    """
    if tracked: track=[]
    uk,vk = u_init,v_init
    zuk = zvk = xiuk = xivk = None

    t = time.time()
    for s in range(n_steps_ACS):
        uk_,vk_ = uk,vk
        t_ = t
        #print('now for u update')
        uk,zuk,xiuk = lADMM_steps(vk,X,X_,Y,a0=uk,z0=zuk,xi0=xiuk,tau=tau,lam=lam,n_steps = n_steps_ADMM)
        #print('now for v update')
        vk,zvk,xivk = lADMM_steps(uk,Y,Y_,X,a0=vk,z0=zvk,xi0=xivk,tau=tau,lam=lam,n_steps = n_steps_ADMM)

        t = time.time()
        if tracked:
            track.append(
            {'t':t-t_,'uk':uk_,'zuk':zuk,'xiuk':xiuk,'vk':vk_,'zvk':zvk,'xivk':xivk}
            )

    if tracked: return track
    else: return [uk,vk]


def suo_first_lazy(X,Y,tau=0.01,lam=3):
    """
    Say this is lazy because we only go for a fixed number of ACS/ADMM steps rather than to convergence
    """
    start = time.time()

    u_init,v_init = suo_init(X,Y)
    uf,vf = next_ccs_lazy(u_init,v_init,X,Y,X,Y,tau=tau,n_steps_ACS=12,n_steps_ADMM=100,lam=lam,tracked=False)

    end = time.time()
    elapsed = end-start
    #print ("Time elapsed:", elapsed) #for debugging

    return uf.reshape(-1,1),vf.reshape(-1,1),elapsed

def first_K_ccs_lazy(X,Y,tau_list,n_steps_ACS=10,n_steps_ADMM=200,lam=10):
    """
    Say this is lazy because we only go for a fixed number of ACS/ADMM steps rather than to convergence
    """
    m,p = X.shape[1],Y.shape[1]
    Uest = np.zeros((m,0))
    Vest = np.zeros((p,0))
    K = len(tau_list)
    U_init,V_init = utils.suo_init_K(X,Y,K=K)
    start = time.time()

    for k in range(K):
        tau = tau_list[k]
        print(f"Getting {k}th ccs with tau={tau}")

        X_ = np.block([[X],[Uest.T @ X.T @ X]])
        Y_ = np.block([[Y],[Vest.T @ Y.T @ Y]])

        u_init,v_init = U_init[:,k],V_init[:,k]
        uf,vf = next_ccs_lazy(u_init,v_init,X,Y,X_,Y_,\
        n_steps_ACS=n_steps_ACS, n_steps_ADMM=n_steps_ADMM, lam=lam,tau=tau)

        Uest = np.block([Uest,uf.reshape(-1,1)])
        Vest = np.block([Vest,vf.reshape(-1,1)])

    end = time.time()
    elapsed = end-start
    print ("Time elapsed:", elapsed)

    return Uest,Vest,elapsed
