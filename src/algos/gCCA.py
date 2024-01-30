import numpy as np
from utils import cca_from_cov_mat, isPSD, get_near_psd
from sklearn.covariance import graphical_lasso
from gglasso.problem import glasso_problem
import time

def sb_algo(X,Y,**kwargs):
    m=X.shape[1]
    #convert to joint data
    Z = np.block([X,Y])
    emp_cov = Z.T @ Z
    #one should probably care about scaling, but we only care about directions so not so important for now

    #estimate joint covariance
    emp_lasso = graphical_lasso(emp_cov,**kwargs)
    est_Sig = emp_lasso[0]
    Uh,Vh,Dh = cca_from_cov_mat(est_Sig,m)
    #here h denotes hat for estimate

    return (Uh,Vh,Dh)

def sb_first_cc(X,Y,alpha,mode='cd',**kwargs):
    start=time.time()

    Usb,Vsb,Dsb = sb_algo(X,Y,alpha=alpha,mode=mode,**kwargs)
    uf = Usb[:,0]
    vf = Vsb[0,:]

    end = time.time()
    elapsed = end-start
    print ("Time elapsed:", elapsed)
    return (uf.reshape(-1,1),vf.reshape(-1,1),elapsed)

def sb_algo_orig_scale(X,Y,**kwargs):
    n,m=X.shape[:2]; p = Y.shape[1]
    #convert to joint data
    Z = np.block([X,Y])
    emp_cov = Z.T @ Z/n

    #estimate joint covariance
    try:
        emp_lasso = graphical_lasso(emp_cov,**kwargs)
        est_Sig = emp_lasso[0]
        Uh,Vh,Dh = cca_from_cov_mat(est_Sig,m)
        #here h denotes hat for estimate
    except FloatingPointError:
        print('Floating Point Error!')
        Uh = np.zeros((m,m))*np.nan; Vh = np.zeros((p,p))*np.nan
        Dh = np.zeros(min(m,p))*np.nan
    return (Uh,Vh,Dh)


###################
def sb_algo_glasso(X,Y,pen=0.05,max_iter=100,pdef_handling=None):
    return sb_algo_glasso_gen(X,Y,pen=pen,max_iter=max_iter,pdef_handling=pdef_handling,gam=None)

def sb_algo_glasso_gen(X,Y,pen=0.05,max_iter=100,pdef_handling=None,gam=None):
    """
    note Vh columns are now estimates!!!
    """
    m = X.shape[1]
    N = X.shape[0]
    #convert to joint data
    Z = np.block([X,Y])
    emp_cov = Z.T @ Z / N

    if gam is not None:
        latent = True
        reg_params = {'lambda1':pen*gam,'mu1':pen}
    else:
        latent = False
        reg_params = {'lambda1': pen}

    P = glasso_problem(emp_cov, N, reg_params = reg_params, latent = latent, do_scaling = False)
    P.solve(solver_params={'max_iter':max_iter})
    sol_om = P.solution.precision_
    if latent: sol_om = P.solution.precision_ - P.solution.lowrank_
    est_Sig = np.linalg.inv(sol_om)

    if not isPSD(sol_om):
        print('not positive definite; so regularising...')
        if pdef_handling == 'near_sig_psd':
            est_Sig = get_near_psd(est_Sig)
        elif pdef_handling == 'near_om_psd':
            sol_om = get_near_psd(sol_om)
            est_Sig = np.linalg.inv(sol_om)
        elif pdef_handling == '4x_iters':
            # factor of 4 is arbitrary but probably sufficient and not overkill
            if not isPSD(sol_om):
                P.solve(solver_params={'max_iter':4*max_iter})
                sol_om = P.solution.precision_
                est_Sig = np.linalg.inv(sol_om)
        else: print(f'unknown error handling {pdef_handling}')

    try:
        Uh,Vh,Dh = cca_from_cov_mat(est_Sig,m,zero_cut_off=10**-8)
    except np.linalg.LinAlgError:
        # changed here because seemed to be case that matrix passed isPD check then failed SVD...
        print('Linear Algebra Error with ccs...!!!!...')
        if pdef_handling == 'near_sig_psd':
            est_Sig = get_near_psd(est_Sig)
        elif pdef_handling == 'near_om_psd':
            sol_om = get_near_psd(sol_om)
            est_Sig = np.linalg.inv(sol_om)
        elif pdef_handling == '4x_iters':
            # factor of 4 is arbitrary but probably sufficient and not overkill
            if not isPSD(sol_om):
                P.solve(solver_params={'max_iter':4*max_iter})
                sol_om = P.solution.precision_
                est_Sig = np.linalg.inv(sol_om)
        else: print(f'unknown error handling {pdef_handling}')
        Uh,Vh,Dh = cca_from_cov_mat(est_Sig,m,zero_cut_off=10**-8)
    #here h denotes hat for estimate

    return (Uh,Vh,Dh)
