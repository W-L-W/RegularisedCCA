import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import sqrtm
from sklearn.preprocessing import normalize
import time

import src.utils as utils


#Prerequisite functions and then main code for single ADMM step
def bigproxmf(x,mu,c,tau):
    ind1 = (x+mu*c > mu*tau)
    ind2 = (x+mu*c < -mu*tau)
    return ind1*(x+mu*c - mu*tau) + ind2*(x+mu*c + mu*tau)

def proxlg(x):
    mag = np.linalg.norm(x,ord=2)
    if (mag>1): return x/mag
    else: return x


## Fixed number of ADMM steps...
def lADMM_steps(b,C,C_,D,a0,z0=None,xi0=None,tau=0.001,lam=10,n_steps=20,tracked=False):
    """
    Use ADMM for a single optimisation of the biconvex objective for finding (r+1)th canonical vector
    We identify (a,b,C,D) with the (u,v,X,Y) from algorithm in appendix of Suo's paper
    (So treat b as fixed and find best a)
    In this formulation, the duals are most important - so will explicitly make them main arguments
    """
    # we have an auxilliary matrix I_top which helps us encode the orthogonality constraints
    n = C.shape[0]
    r = C_.shape[0] - n
    I_top = np.block([[np.identity(n)],[np.zeros([r,n])]])

    # define tolerances - these update as our estimates update (estimates can change a lot)
    p = C.shape[1]

    # set the variable proximal operator
    c = C.transpose()@D@b
    mu=lam/(2*np.linalg.norm(C_,ord=2)**2) # (numpy operator norm)
    proxmf = (lambda x: bigproxmf(x,mu,c,tau))

    # initialise: if have no duals to start with the below gives sensible default
    ak  = a0
    if (z0 is None) or (xi0 is None):
        zk  = proxlg(C@ak)
        xik = C_@ak-I_top@zk
    else:
        zk = z0
        xik = xi0

    #tracked is a tool for debugging - just puts relevant step data into a list
    if tracked: track = []

    #main ADMM loop
    for _ in range(n_steps):
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


def next_ccs_lazy(u_init, v_init, X, Y, X_, Y_,
                  n_steps_ACS=10,n_steps_ADMM=100, lam=3,tau=0.01,tracked=False
    ):
    """
    Say this is lazy because we only go for a fixed number of ACS/ADMM steps rather than to convergence
    """
    if tracked: track=[]
    uk,vk = u_init,v_init
    zuk = zvk = xiuk = xivk = None

    t = time.time()
    for _ in range(n_steps_ACS):
        uk_,vk_ = uk,vk
        t_ = t

        uk,zuk,xiuk = lADMM_steps(vk,X,X_,Y,a0=uk,z0=zuk,xi0=xiuk,tau=tau,lam=lam,n_steps = n_steps_ADMM)
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

    u_init,v_init = utils.suo_init(X,Y)
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
