import numpy as np
from typing import Iterable

from real_data.loading import get_dataset
from src.scaffold.core import Data, CV, MVNData, MVNCV, MVNFactory
from src.utils import covs

# COORDINATION
##############
def get_cv_object(dataset: str, algo: str) -> CV:
    """ Inputs are strings representing the dataset and algorithm respectively"""
    data = get_dataset(dataset)
    cv_obj = CV(data,folds=5,algo=algo,K=10)
    return cv_obj

def get_cv_obj_from_data(data: Data, algo: str) -> CV: #MVN=False
    """ For all simulations we use 5 folds and top 10 directions; this helper function abstracts that out """
    if type(data) == MVNData: # if dataset is generated from known multivariate normal distribution
        return MVNCV(data,folds=5,algo=algo,K=10)
    else:
        return CV(data,folds=5,algo=algo,K=10)
    
def compute_everything(cv_obj: CV, pens: Iterable[float]):
    cv_obj.fit_algo(pens)
    if type(cv_obj) == MVNCV:
        cv_obj.process_oracle()
    cv_obj.process_cv()
    cv_obj.process_stab()
    

# PURELY SYNTHETIC DATA
################
def cov_type_to_fac(cov_type: str):
    """convert from string descriptor to cov mat for Multivariate Normal (MVN)"""
    # mac is short for machine (working in the factory)
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