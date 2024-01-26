import numpy as np
import time
import os

from src.algos import sb_algo_glasso, first_K_ccs_lazy, PMD_CCA
from cca_zoo.models import rCCA,SCCA_IPLS

from src.scaffold.core import MVN

# IO behaviour
dir_path = os.path.dirname(os.path.realpath(__file__))
base_ds = dir_path+'/../big_sims/oop_sims'

def mvn_folder_name(mvn: MVN,rs: int,n: int):
    cov_desc,p,q = mvn.cov_desc,mvn.p,mvn.q
    folder_name = base_ds + f'/{cov_desc}/p{p}q{q}n{n}/rs{rs}'
    return folder_name


# Algorithms: from string algorithm names to estimates
def get_ests_n_time(X,Y, algo: str, pen: float, K: int):
    n = X.shape[0]
    Xtil,Ytil = X/np.sqrt(n),Y/np.sqrt(n)
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
    return Ue[:,:K],Ve[:,:K],te