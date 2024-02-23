import numpy as np
import time

from cca_zoo.models import rCCA,SCCA_IPLS
from src.algos import sb_algo_glasso, first_K_ccs_lazy, PMD_CCA

algo_labels = {'wit':'sPLS',
               'suo': 'sCCA',
               'gglasso': 'gCCA',
               'ridge': 'rCCA',}

# from string algorithm names to estimates
def get_ests_n_time(X,Y, algo: str, pen: float, K: int):
    n = X.shape[0]
    Xtil, Ytil = X/np.sqrt(n), Y/np.sqrt(n)
    start = time.time()

    if algo == 'gglasso':
        (Ue,Ve,De) = sb_algo_glasso(X,Y,pen=pen,pdef_handling='near_om_psd')
    elif algo == 'gglasso300':
            (Ue,Ve,De) = sb_algo_glasso(X,Y,pen=pen,max_iter=300)
    elif algo == 'suo':
        taus = [pen]*K
        Ue,Ve,te0 = first_K_ccs_lazy(Xtil,Ytil,taus)
    elif algo == 'wit':
        cs = [pen]*K
        Ue,Ve,De,te0 = PMD_CCA(Xtil,Ytil,cs)
    elif algo == 'ridge':
        try:
            model = rCCA(c=[pen,pen],latent_dims=K)
            model.fit([X,Y])
            Ue,Ve = model.weights
        except: # previously had LinAlgError and ValueError:
            Ue,Ve = np.zeros((X.shape[1],K)),np.zeros((Y.shape[1],K))
    elif algo == 'IPLS':
        try:
            model = SCCA_IPLS(c=[pen,pen], random_state=0,latent_dims=K)
            model.fit([X,Y])
            Ue,Ve = model.weights
        except: # previously had LinAlgError and ValueError:
            Ue,Ve = np.zeros((X.shape[1],K)),np.zeros((Y.shape[1],K))
    else:
        raise(f'Algorithm {algo} not implemented yet!')
    te = time.time()-start
    return Ue[:,:K], Ve[:,:K], te

# choosing penalties depends primarily on algorithm, and dataset size
# these choices worked well for the three datasets we considered, and the setting of our study 
# but one may want to edit the choice of penalties for other datasets or applications
def get_pens(algo: str, n: int, mode: str ='run'):
    """
    Determine penalties to use for given algo as a function of size of data
    Run-time will scale linearly with number of different penalties
    mode: 'run' or 'debug'
    """
    if mode == 'debug':
        if algo == 'wit': pens = np.logspace(np.log10(2),np.log10(4),2)
        elif algo in ['gglasso', 'suo']: pens = np.logspace(-3, 3, 2)
        elif algo == 'ridge': pens = np.array([10**-3, 1 - 10**-3])
    elif mode == 'run':
        if algo == 'wit': 
            pens = np.logspace(np.log10(1),np.log10(7),15)
        elif (algo == 'suo') or (algo == 'gglasso') or (algo == 'IPLS'):
            def pen_poly(n): return 2 * (n ** -0.5)
            pens = np.block([np.logspace(-20,-3,2),np.logspace(-2,-1.2,3),np.logspace(-1,1,16),np.logspace(1.2,2,4)]) * pen_poly(n) 
        elif (algo == 'ridge'):
            # changed 19 Jan to have more coverage near 1
            pens_from_zero = np.block([np.logspace(-20,-4,2),np.logspace(-4,-1,6),np.linspace(0.2,0.8,6)])   
            pens_to_one = 1 - np.logspace(-1, -4, 4)
            pens = np.block([pens_from_zero,pens_to_one])  
    else:
        raise ValueError(f'Unknown mode {mode} for get_pens')
    return pens