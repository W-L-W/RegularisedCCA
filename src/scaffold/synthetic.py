# Think intention here was to copy over the block of code from oop_utils starting with 'Setting' class
# TODO work through that!
from src.scaffold.wrappers import get_dataset
from src.algos import first_K_ccs_lazy
from src.utils import gram_schmidt
import numpy as np
import os

from real_data.loading import pboot_filename

# and dictionary to determine behaviour of parametric bootstrap
# currently copied over manually from previous experiment runs
# TODO: have a more systematic way to determine penalty settings
# for now used factor of 3 smaller penalty than cv optimal
bootstrap_params = {
    ('nutrimouse','ridge','cvm3'): {'ridge': 0.066},
    ('nutrimouse', 'gglasso', 'cvm3'): {'pen': 0.0066},
    ('nutrimouse', 'suo_ridge', 'cvm3'): {'pen': 0.026, 'K': 10, 'ridge': 0.01},
}
# note still to be finished - need to implement the main function before doing any more like this

def save_pboot_cov(dataset:str, regn:str, regn_mode:str):
    Sig = gen_parametric_bootstrap_cov(dataset, regn, regn_mode)
    filename = pboot_filename(dataset, f'{regn}_{regn_mode}_cov')
    # make directory if it doesn't exist already
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(filename, Sig)

def load_pboot_cov(dataset:str, regn:str, regn_mode:str):
    filename = pboot_filename(dataset, f'{regn}_{regn_mode}_cov')
    Sig = np.load(filename)
    return Sig

def gen_parametric_bootstrap_cov(dataset: str, regn='ridge', regn_mode = 'cv'):
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
    return Sig

if __name__ == '__main__':
    save_pboot_cov('nutrimouse', 'gglasso', 'cvm3')