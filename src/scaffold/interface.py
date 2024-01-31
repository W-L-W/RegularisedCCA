import numpy as np

import time
import os

from cca_zoo.models import rCCA,SCCA_IPLS

from src.algos import sb_algo_glasso, first_K_ccs_lazy, PMD_CCA
from src.utils import covs
from src.scaffold.core import Data, CV, MVNData, MVNCV, MVNFactory
from src.scaffold.synthetic import gen_parametric_bootstrap_cov

from real_data.loading import get_breastdata, get_nutrimouse, get_microbiome, pboot_filename


# IO behaviour
dir_path = os.path.dirname(os.path.realpath(__file__))
base_ds = dir_path+'/../expmts_out'

def mvn_folder_name(mvn: MVNData, rs: int, n: int):
    cov_desc,p,q = mvn.cov_desc, mvn.p, mvn.q
    folder_name = base_ds + f'/{cov_desc}/p{p}q{q}n{n}/rs{rs}/'
    return folder_name


# COORDINATION
##############
def get_cv_object(dataset: str, algo: str) -> CV:
    """ Inputs are strings representing the dataset and algorithm respectively"""
    data = get_dataset(dataset)
    cv_obj = CV(data,folds=5,algo=algo,R=10)
    return cv_obj

def get_cv_obj_from_data(data: Data, algo: str) -> CV: #MVN=False
    """ For all simulations we use 5 folds and top 10 directions; this helper function abstracts that out """
    if type(data) == MVNData: # if dataset is generated from known multivariate normal distribution
        return MVNCV(data,folds=5,algo=algo,R=10)
    else:
        return CV(data,folds=5,algo=algo,R=10)


# ALGORITHMS
############
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
    return Ue[:,:K],Ve[:,:K],te

# choosing penalties depends primarily on algorithm, and dataset size
# these choices worked well for the three datasets we considered, and the setting of our study 
# but one may want to edit the choice of penalties for other datasets or applications
def get_pens(algo,data):
    """
    Determine penalties to use for given algo as a function of size of data
    Run-time will scale linearly with number of different penalties
    """
    if algo == 'wit': 
        pens = np.logspace(np.log10(1),np.log10(7),15)
    elif (algo == 'suo') or (algo == 'gglasso') or (algo == 'IPLS'):
        def pen_poly(n): return 2 * (n ** -0.5)
        pens = np.block([np.logspace(-20,-3,2),np.logspace(-2,-1.2,3),np.logspace(-1,1,16),np.logspace(1.2,2,4)]) * pen_poly(data.n) 
    elif (algo == 'ridge'):
        # changed 19 Jan to have more coverage near 1
        pens_from_zero = np.block([np.logspace(-20,-4,2),np.logspace(-4,-1,6),np.linspace(0.2,0.8,6)])   
        pens_to_one = 1 - np.logspace(-1, -4, 4)
        pens = np.block([pens_from_zero,pens_to_one])  
    return pens

# REAL DATA
###########
def get_dataset(dataset: str):
    """ Options: 'breastdata', 'nutrimouse', 'microbiome' """
    if dataset=='breastdata':
        data = get_breastdata()
    elif dataset=='nutrimouse':
        data = get_nutrimouse()
    elif dataset == 'microbiome':
        data = get_microbiome()
    else:
        print(f'Unrecognised dataset: {dataset}')
    return data

# and dictionary to determine behaviour of parametric bootstrap
# currently copied over manually from previous experiment runs
# TODO: have a more systematic way to determine penalty settings
bootstrap_params = {
    ('nutrimouse','ridge','cv'): {'ridge': 0.1},
    ('nutrimouse', 'gglasso', 'cv'): {'pen': 0.1},
    ('nutrimouse', 'suo_ridge', 'cv'): {'pen': 0.1, 'K': 10, 'ridge': 0.01},
}
# note still to be finished - need to implement the main function before doing any more like this

def save_pboot_cov(dataset:str, regn:str, regn_mode:str):
    Sig = gen_parametric_bootstrap_cov(dataset, regn, regn_mode)
    filename = pboot_filename(dataset, f'{regn}_{regn_mode}_cov.csv')
    np.save(filename, Sig)

def load_pboot_cov(dataset:str, regn:str, regn_mode:str):
    filename = pboot_filename(dataset, f'{regn}_{regn_mode}_cov.csv')
    Sig = np.load(filename)
    return Sig


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