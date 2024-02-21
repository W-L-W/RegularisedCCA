# All functions building on top of core.py and used for generating or analysing synthetic data
import numpy as np
import pandas as pd
import os

from src.scaffold.wrappers import cov_type_to_fac, compute_everything
from src.scaffold.core import MVNCV, MVNDist, MVNFactory
from src.utils import covs
from src.utils import gram_schmidt
from src.algos import first_K_ccs_lazy


from real_data.loading import get_dataset, pboot_filename


# PARAMETRIC BOOTSTRAP
######################

# dictionary to determine behaviour of parametric bootstrap
bootstrap_params = {
    ('nutrimouse','ridge','cvm3'): {'ridge': 0.066},
    ('nutrimouse', 'gglasso', 'cvm3'): {'pen': 0.0066},
    ('nutrimouse', 'suo_ridge', 'cvm3'): {'pen': 0.026, 'K': 10, 'ridge': 0.01},
    ('microbiome', 'ridge', 'cv'): {'ridge': 0.11},
    ('microbiome', 'gglasso', 'cv'): {'pen': 0.0059},
    ('microbiome', 'suo_ridge', 'cv'): {'pen': 0.013, 'K': 10, 'ridge': 0.01}
}
# currently copied over manually from previous experiment runs
# TODO: have a more systematic way to determine penalty settings
# for now used factor of 3 smaller penalty than cv optimal

def save_pboot_cov(dataset:str, regn:str, regn_mode:str):
    p, Sig = gen_parametric_bootstrap_cov(dataset, regn, regn_mode)
    filename = pboot_filename(dataset, f'{regn}_{regn_mode}_cov.npz')
    # make directory if it doesn't exist already
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez(filename, p=p, Sig=Sig)

def load_pboot_mvn(dataset:str, regn:str, regn_mode:str):
    filename = pboot_filename(dataset, f'{regn}_{regn_mode}_cov.npz')
    # if file does not exist, need to generate it first
    if not os.path.exists(filename):
        print('Covariance matrix file does not exist, so generating it now')
        save_pboot_cov(dataset, regn, regn_mode)
    npzfile = np.load(filename)
    p, Sig = npzfile['p'], npzfile['Sig']
    return MVNDist(Sig,p,cov_desc = f'pboot/{dataset}/{regn}_{regn_mode}')

def gen_parametric_bootstrap_cov(dataset: str, regn='ridge', regn_mode = 'cv'): # -> Tuple[int, PSDMatrix]
    print('loading dataset')
    data = get_dataset(dataset)
    print('loaded dataset')
    def demean_cols(M): return M - M.mean(axis=0)
    X,Y = list(map(demean_cols,[data.X,data.Y]))
    n,p = X.shape[:2]; q = Y.shape[1]
    #convert to joint data
    Z = np.block([X,Y])
    emp_cov = Z.T @ Z/n

    # get penalty
    params = bootstrap_params[(dataset,regn,regn_mode)]

    # estimate regularised covariance
    if regn == 'ridge':
        ridge_param = params['ridge']
        Sig = emp_cov +  ridge_param * np.identity(p+q)
    elif regn == 'gglasso':
        print('fitting glasso problem')
        from gglasso.problem import glasso_problem
        pen_param = params['pen']
        reg_params = {'lambda1': pen_param}
        P = glasso_problem(emp_cov, n, reg_params = reg_params, latent = False, do_scaling = False)
        P.solve(solver_params={'max_iter':100})
        print('solved glasso problem')
        Sig = np.linalg.inv(P.solution.precision_)
    elif regn == 'suo_ridge':
        Xtil,Ytil = X/np.sqrt(n),Y/np.sqrt(n)
        pen_param, K, ridge_param = params['pen'], params['K'], params['ridge']
        taus = [pen_param]*K
        Ue,Ve,te0 = first_K_ccs_lazy(Xtil,Ytil,taus)
        De = np.diag(np.diag(Ue.T @ Xtil.T @ Ytil @ Ve)) # extract diagonal elements
        print(De)
        # np.sqrt(np.log(p) / n) * np.identity(p)
        Sigxx = Xtil.T @ Xtil +  ridge_param
        Sigyy = Ytil.T @ Ytil +  ridge_param

        #(Ue.T @ Xtil.T @ Ytil @ Ve)[:8,:8].round(2) # to check looks reasonable
        Ue = gram_schmidt(Ue,Sigxx)
        Ve = gram_schmidt(Ve,Sigyy)

        Sigxy = Sigxx @ Ue @ De @ Ve.T @ Sigyy
        Sigyx = Sigxy.T
        Sig = np.block([[Sigxx,Sigxy],[Sigyx,Sigyy]])
    else:
        raise NotImplementedError()
    
    print('on to save')
    return p, Sig


# CONVENIENT STRUCT FOR SYNTHETIC DATA
#######################################
# and utility functions exploiting this struct
class Setting():
    def __init__(self,cov_type,algo,p,q,n,folds,K): #,rss
        self.cov_type = cov_type
        self.algo = algo
        self.p = p
        self.q = q
        self.n = n
        self.folds = folds
        self.K = K
        #self.rss = rss
        self.tuple = (cov_type,algo,p,q,n,folds,K)

    def __str__(s):
        return f'cov: {s.cov_type}, algo: {s.algo}, (p,q,n): {(s.p,s.q,s.n)}, folds: {s.folds}, K: {s.K}'#', rss: {s.rss}'

def load_mvn_cv(setting,rs):
    (cov_type,algo,p,q,n,folds,K) = setting.tuple
    fac = cov_type_to_fac(cov_type)
    mvn = fac.build_mvn(p,q)
    data = mvn.gen_data(rs,n)
    cv_obj = MVNCV(data,folds,algo,K)
    return cv_obj

def compute_everything_single_seed(setting,rs,pen_list):
    """Well everything I care about for now..."""
    cv_obj = load_mvn_cv(setting,rs)
    compute_everything(cv_obj, pen_list)

def load_metric_dfs(setting,rs):
    cv_obj = load_mvn_cv(setting,rs)
    df_full = cv_obj.load_df_oracle()
    df_cvav = cv_obj.load_summary('cv averages')
    return df_full, df_cvav

def best_pen(df,obj_string,best='min'):
    if best=='min':
        return df.loc[df[obj_string].argmin(),'pen']
    elif best=='max':
        return df.loc[df[obj_string].argmax(),'pen']

# Key improvement over previous version is the ability to merge the full and cv versions
def best_pen_rows(setting,rs,pen_objs):
    df_full,df_cvav = load_metric_dfs(setting,rs)
    df_cv_means = df_cvav.xs('mean',axis=1, level=1, drop_level=True).reset_index()
    
    # combine into a single df which we give short name for convenience...
    df = pd.merge(df_full,df_cv_means,on='pen',suffixes=['','_cv'])

    # collect dict of best pens:
    pen_dict = {objv: best_pen(df,objv,sign) for objv,sign in pen_objs.items()}
    ## print(pen_dict['rho1'],pen_dict['rho1_cv']) #for debugging

    #select corresponding rows
    rows = [df[df['pen']==pen_dict[objv]] for objv in pen_objs.keys()]
    df = pd.concat(rows)
    df['pen_obj'] = pen_objs.keys()
    df['min_or_max'] = pen_objs.values()
    return df

def av_best_pen_rows(setting,rss,pen_objs):
    single_dfs = [best_pen_rows(setting,rs,pen_objs) for rs in rss]
    df_comb = pd.concat(single_dfs)
    df = df_comb.groupby(['pen_obj','min_or_max']).agg([np.mean,np.std]).reset_index()

    for s,v in zip(['p','q','n'],[setting.p,setting.q,setting.n]):
        df[s]=v
    return df

def select_values(df_best_rows,met):
    """met is string of what metric you want to plot"""
    vals = [(met,'mean'),(met,'std')]
    df = df_best_rows.pivot(index='n',columns='pen_obj',values=vals)
    df.columns = df.columns.set_levels(['mean','std'],level=0).set_names(met,level=0)
    return df

def vary_n_best_rows(algo,cov_type,p,q,folds,K,ns,rss,pen_objs):
    settings = [Setting(cov_type,algo,p,q,n,folds,K) for n in ns]

    comb_best_rows = [av_best_pen_rows(setting,rss,pen_objs) for setting in settings]

    return pd.concat(comb_best_rows)



# PURELY SYNTHETIC DATA GENERATION
##################################
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