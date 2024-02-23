# All functions building on top of core.py and used for generating or analysing synthetic data
import numpy as np
import pandas as pd
import os
from typing import Iterable

from src.scaffold.wrappers import compute_everything
from src.scaffold.core import MVNCV, MVNDist, MVNFactory
from src.utils import covs
from src.utils import gram_schmidt
from src.algos import first_K_ccs_lazy


from real_data.loading import get_dataset, pboot_filename, dset_abbrev, dset_abbrev_inv


# PARAMETRIC BOOTSTRAP
######################
reg_abbrev = {
    'gglasso': 'ggl',
    'ridge': 'rdg',
    'suo_ridge': 'suoR'
}
reg_abbrev_inv = {v:k for k,v in reg_abbrev.items()}

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

# for converting between string identifiers and three defining arguments
def pboot_params_to_string(dataset:str, regn:str, param_choice:str):
    return f'pboot_{dset_abbrev[dataset]}_{reg_abbrev[regn]}_{param_choice}'

def pboot_string_to_params(pboot_string:str):
    _, dset, regn, param_choice = pboot_string_to_params(pboot_string)
    return dset_abbrev_inv[dset], reg_abbrev_inv[regn], param_choice


def save_pboot_cov(dataset:str, regn:str, param_choice:str):
    """To determine where the regularised cov mat is saved as npz file for quick loading later"""
    p, Sig = gen_parametric_bootstrap_cov(dataset, regn, param_choice)
    filename = pboot_filename(dataset, f'{regn}_{param_choice}_cov.npz')
    # make directory if it doesn't exist already
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez(filename, p=p, Sig=Sig)

def load_pboot_mvn(dataset:str, regn:str, param_choice:str):
    filename = pboot_filename(dataset, f'{regn}_{param_choice}_cov.npz')
    # if file does not exist, need to generate it first
    if not os.path.exists(filename):
        print('Covariance matrix file does not exist, so generating it now')
        save_pboot_cov(dataset, regn, param_choice)
    npzfile = np.load(filename)
    p, Sig = npzfile['p'], npzfile['Sig']
    return MVNDist(Sig, p, path_stem = f'pboot/{dataset}/{regn}_{param_choice}')

def gen_parametric_bootstrap_cov(dataset: str, regn='ridge', param_choice = 'cv'): # -> Tuple[int, PSDMatrix]
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
    params = bootstrap_params[(dataset,regn,param_choice)]

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

def pboot_setting(pboot_string: str, algo: str, n: int, folds: int, K: int):
    dataset, regn, param_choice = pboot_string_to_params(pboot_string)
    mvn = load_pboot_mvn(dataset, regn, param_choice)
    return Setting(mvn, algo, n, folds, K)



def synth_setting(cov_type: str, algo: str, p: int, q: int, n: int, folds: int, K: int):
    pass
# to implement once re-hashed the cov fac idea

class Setting():
    def __init__(s, mvn: MVNDist, algo: str, n: int, folds: int, K: int):
        s.mvn = mvn
        s.algo = algo
        s.n = n
        s.folds = folds
        s.K = K
        s.tuple = (mvn, algo, n, folds, K)

    def __str__(s):
        return f'cov: {s.cov_type}, algo: {s.algo}, (p,q,n): {(s.p,s.q,s.n)}, folds: {s.folds}, K: {s.K}'#', rss: {s.rss}'


class PenObjs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Validate that all values are either 'min' or 'max'
        assert all(value in ['min', 'max'] for value in self.values()), \
            "All values must be 'min' or 'max'."
    def __setitem__(self, key, value):
        # Ensure only 'min' or 'max' can be set as values
        if value not in ['min', 'max']:
            raise ValueError("Value must be 'min' or 'max'.")
        super().__setitem__(key, value)

    
def one_seed_cv_obj(setting: Setting, rs: int):
    (mvn, algo, n, folds, K) = setting.tuple
    data = mvn.gen_data(rs,n)
    cv_obj = MVNCV(data,folds,algo,K)
    return cv_obj

def one_seed_fit(setting: Setting ,rs: int, pen_list: Iterable[float]):
    cv_obj = one_seed_cv_obj(setting,rs)
    compute_everything(cv_obj, pen_list)

def load_metric_dfs(setting: Setting, rs: int):
    cv_obj = one_seed_cv_obj(setting,rs)
    df_full = cv_obj.load_df_oracle()
    df_cvav = cv_obj.load_summary('cv averages')
    return df_full, df_cvav

def best_pen(df: pd.DataFrame, objv: str, min_or_max='min'):
    """Return the penalty that {'min','max'}imises the given objv column of df
    df must have columns: 'pen' and objv """
    if min_or_max=='min':
        return df.loc[df[objv].argmin(),'pen']
    elif min_or_max=='max':
        return df.loc[df[objv].argmax(),'pen']

# Key improvement over previous version is the ability to merge the full and cv versions
def best_pen_rows(setting: Setting, rs: int, pen_objs: PenObjs):
    df_full,df_cvav = load_metric_dfs(setting,rs)
    df_cv_means = df_cvav.xs('mean',axis=1, level=1, drop_level=True).reset_index()
    
    # combine into a single dataframe, used for remainder of the function
    df = pd.merge(df_full,df_cv_means,on='pen',suffixes=['','_cv'])

    # collect dict of best pens:
    pen_dict = {objv: best_pen(df,objv,sign) for objv,sign in pen_objs.items()}

    #select corresponding rows
    rows = [df[df['pen']==pen_dict[objv]] for objv in pen_objs.keys()]
    df = pd.concat(rows)
    df['pen_obj'] = pen_objs.keys()
    df['min_or_max'] = pen_objs.values()
    return df

def av_best_pen_rows(setting: Setting ,rss: Iterable[int], pen_objs: PenObjs):
    single_dfs = [best_pen_rows(setting,rs,pen_objs) for rs in rss]
    df_comb = pd.concat(single_dfs)
    df = df_comb.groupby(['pen_obj','min_or_max']).agg([np.mean,np.std]).reset_index()

    for s,v in zip(['p','q','n'],[setting.p,setting.q,setting.n]):
        df[s]=v
    return df

def vary_n_best_rows(algo: str, mvn: MVNDist, folds: int, K: int, ns: Iterable[int], rss: Iterable[int], pen_objs: PenObjs):
    settings = [Setting(mvn,algo,n,folds,K) for n in ns]

    comb_best_rows = [av_best_pen_rows(setting,rss,pen_objs) for setting in settings]

    return pd.concat(comb_best_rows)

def select_values(df_best_rows,met):
    """met is string of what metric you want to plot"""
    vals = [(met,'mean'),(met,'std')]
    df = df_best_rows.pivot(index='n',columns='pen_obj',values=vals)
    df.columns = df.columns.set_levels(['mean','std'],level=0).set_names(met,level=0)
    return df



# PURELY SYNTHETIC DATA GENERATION
##################################
def synth_mvn(cov_desc: str, p: int, q: int):
    """Return MVNDist object for a given covariance type and dimension"""
    if cov_desc == '3spike_id':
        Sig = covs.cov_from_three_spikes(p,q,method='identity')
    elif cov_desc == '3spike_to':
        mac = lambda p,q: covs.cov_from_three_spikes(p,q,method='toeplitz')
    elif cov_desc == '3spike_sp':
        Sig = covs.cov_from_three_spikes(p,q,method='sparse')
    elif cov_desc == '3spike_sp_sep':
        Sig = covs.cov_from_three_spikes(p,q,method='sparse',d1=0.95,d2=0.5,d3=0.4)
    elif cov_desc == '3spike_sp_old':
        Sig = covs.cov_from_three_spikes(p,q,method='sparse',d1=0.95,d2=0.5,d3=0.4)
    elif cov_desc == 'suo_id_unif':
        Sig = covs.suo_basic('identity',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_to_unif':
        Sig = covs.suo_basic('toeplitz',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_sp_unif':
        Sig = covs.suo_basic('sparse',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_id_rand':
        Sig = covs.suo_rand('identity',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_to_rand':
        Sig = covs.suo_rand('toeplitz',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_sp_rand':
        Sig = covs.suo_rand('sparse',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_lr_rand':
        Sig = covs.suo_rand('low_rank',p,q,rho=0.9,ms=5,ps=5)
    elif cov_desc == 'suo_sp_rand50':
        Sig = covs.suo_rand('sparse',p,q,rho=0.9,ms=50,ps=50)
    elif cov_desc == 'erdos':
        from gglasso.helper.data_generation import generate_precision_matrix
        p_tot = p + q
        Sig, _ = generate_precision_matrix(p=p_tot, M=1, style='erdos', prob=0.1, seed=1234)
    elif cov_desc == 'powerlaw':
        from gglasso.helper.data_generation import generate_precision_matrix
        p_tot = p + q
        # got mysterious networkx error for m=p=200 when had seed 1234 but OK with seed 1200 so hacky fix...
        nxseed = 1200 if p==200 else 1234
        structuredSig, _ = generate_precision_matrix(p=p_tot, M=1, style='powerlaw',gamma=3, seed=nxseed)
        rng = np.random.default_rng(seed=0)
        perm = rng.permutation(p_tot)
        # in old version I had line below other way round and it did something strange with copy assignment
        Sig = structuredSig[np.ix_(perm,perm)]
    elif cov_desc == '10spikes_size4_sp':
        Sig = covs.geom_corr_decay_sparse_weight_multi_spike(
            p, q, K=10, decay_ratio=0.9, spike_size=4
        )
    elif cov_desc == '8spikes_size3_sp_weighted':
        Sig = covs.geom_corr_decay_sparse_weight_multi_spike(
            p, q, K=8, decay_ratio=0.9, spike_size=3, method='sparse', geom_param=0.9
        )
    elif cov_desc == 'bach_latent':
        (nlatents,decay_ratio,supp_size) = 10,0.9,3
        Sig = covs.cov_from_latents(p,q,nlatents,decay_ratio,supp_size)
    elif cov_desc == 'bach_latent_sp':
        (nlatents,decay_ratio,supp_size) = 10,0.9,3
        Sig = covs.cov_from_latents(p,q,nlatents,decay_ratio,supp_size,method='sparse')
    else:
        raise Exception(f'unrecognised cov_type {cov_desc}- perhaps not yet implemented?')

    return MVNDist(Sig, p, path_stem = f'synth/{cov_desc}/p{p}q{q}')
# to review: the syntax for this saving