import numpy as np

import time
import os

from cca_zoo.models import rCCA,SCCA_IPLS

from src.algos import sb_algo_glasso, first_K_ccs_lazy, PMD_CCA
from src.utils import covs
from src.scaffold.core import MVN, MVNFactory

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
    return Ue[:,:K],Ve[:,:K],te


# Synthetic data generation: from string descriptor to cov mat for Multivariate Normal (MVN):
def cov_type_to_fac(cov_type: str):
    # mac is short for machiine (working in the factory)
    if cov_type == '3spike_id':
        mac = lambda p,q: covs.cov_from_three_spikes(p,q,method='identity')
    elif cov_type == '3spike_to':
        mac = lambda p,q: covs.cov_from_three_spikes(p,q,method='toeplitz')
    elif cov_type == '3spike_sp':
        mac = lambda p,q: covs.cov_from_three_spikes(p,q,method='sparse')
    elif cov_type == '3spike_sp_sep':
        mac = lambda p,q: covs.cov_from_three_spikes(p,q,method='sparse',d1=0.95,d2=0.5,d3=0.4)
    elif cov_type == '3spike_sp_old':
        mac = lambda p,q: covs.cov_from_three_spikes(p,q,method='sparse',d1=0.95,d2=0.5,d3=0.4)
    elif cov_type == 'suo_id_unif':
        mac = lambda p,q: covs.suo_basic('identity',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_to_unif':
        mac = lambda p,q: covs.suo_basic('toeplitz',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_sp_unif':
        mac = lambda p,q: covs.suo_basic('sparse',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_id_rand':
        mac = lambda p,q: covs.suo_rand('identity',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_to_rand':
        mac = lambda p,q: covs.suo_rand('toeplitz',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_sp_rand':
        mac = lambda p,q: covs.suo_rand('sparse',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_lr_rand':
        mac = lambda p,q: covs.suo_rand('low_rank',p,q,rho=0.9,ms=5,ps=5)
    elif cov_type == 'suo_sp_rand50':
        mac = lambda p,q: covs.suo_rand('sparse',p,q,rho=0.9,ms=50,ps=50)
    elif cov_type == 'erdos':
        from gglasso.helper.data_generation import generate_precision_matrix
        def mac(p,q):
            p_tot = p + q
            Sig, Theta = generate_precision_matrix(p=p_tot, M=1, style='erdos', prob=0.1, seed=1234)
            return Sig
    elif cov_type == 'powerlaw':
        from gglasso.helper.data_generation import generate_precision_matrix
        def mac(p,q):
            p_tot = p + q
            # got mysterious networkx error for m=p=200 when had seed 1234 but OK with seed 1200 so hacky fix...
            nxseed = 1200 if p==200 else 1234
            Sig, Theta = generate_precision_matrix(p=p_tot, M=1, style='powerlaw',gamma=3, seed=nxseed)
            rng = np.random.default_rng(seed=0)
            perm = rng.permutation(p_tot)
            # in old version I had line below other way round and it did something strange with copy assignment
            final_Sig = Sig[np.ix_(perm,perm)]
            return final_Sig
    elif cov_type == '10spikes_size4_sp':
        mac = lambda p,q: covs.geom_corr_decay_sparse_weight_multi_spike(
            p, q, K=10, decay_ratio=0.9, spike_size=4
        )
    elif cov_type == '8spikes_size3_sp_weighted':
        mac = lambda p,q: covs.geom_corr_decay_sparse_weight_multi_spike(
            p, q, K=8, decay_ratio=0.9, spike_size=3, method='sparse', geom_param=0.9
        )
    elif cov_type == 'bach_latent':
        def mac(p,q):
            (nlatents,decay_ratio,supp_size) = 10,0.9,3
            Sig = covs.cov_from_latents(p,q,nlatents,decay_ratio,supp_size)
            return Sig
    elif cov_type == 'bach_latent_sp':
        def mac(p,q):
            (nlatents,decay_ratio,supp_size) = 10,0.9,3
            Sig = covs.cov_from_latents(p,q,nlatents,decay_ratio,supp_size,method='sparse')
            return Sig
    else:
        raise Exception(f'unrecognised cov_type {cov_type}- perhaps not yet implemented?')

    return MVNFactory(cov_type, mac)