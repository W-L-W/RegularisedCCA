import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import KFold
from collections import OrderedDict
from typing import List

import src.scaffold.interface as interface
import src.utils as utils

# # A word on motivation:
# This file is messy, and would benefit from refactoring.
# However, I have attempted to maintain certain abstraction barriers to keep modularity
# The key flexibility I want to allow is for a user to apply this scaffold to analyse new datasets and with new CCA algorithms
# while only having to edit code in the interface.py file

# E.g. though this file contains many different ways to save and load data and associated folder name Conventions
# these always build off the folder from a data object
# and the folders for data objects are defined in the interface.py file


# # # Conventions: 
# p, q (problem dims, int), 
# N (sample size, int), 
# K (number of components, int), 
# pen (penalty parameter, float),
# algo (algorithm name, string),


# small utility functions
def ests_to_csv(Ue,Ve,file_name):
    """Saves estimates in form Ue^T Ve^T block concatenation"""
    We = np.block([[Ue],[Ve]])
    data = pd.DataFrame(We.T)
    data.to_csv(file_name,mode='a',index=False,header=False)


class Data():
    def __init__(self,X,Y,folder_name=None,X_labs=None,Y_labs=None):
        assert type(X)==type(Y)==np.ndarray, 'non-array data will mess up indexing'
        self.X = X
        self.Y = Y
        self.p,self.q = X.shape[1],Y.shape[1]
        self.n = X.shape[0]
        if folder_name: self.folder = folder_name
        self.X_labs = X_labs
        self.Y_labs = Y_labs

    def fit_algo(self,algo,pen,K):
        Ue,Ve,te = interface.get_ests_n_time(self.X, self.Y, algo, pen, K)
        return Ue,Ve,te
    

class SolutionPath():
    def __init__(self,data,algo,K):
        self.data = data
        self.algo = algo
        self.K = K
        self.folder = self._create_folder_name
        try:
            _,meta = self.load_ests_n_meta()
            self.pens_fitted = set(meta['pen'])
        except FileNotFoundError:
            self.pens_fitted = set()

    def _create_folder_name(self):
        folder_name = self.data.folder + 'estimates/'
        os.makedirs(folder_name,exist_ok=True)
        return folder_name
    
    def _filename(self, output_type = 'weights'):
        if output_type == 'weights':
            pass
        elif output_type == 'meta data':
            pass # TODO!!!

    def _already_fitted(self,pen):
        return sum([np.isclose(pen,penb,rtol=1e-03, atol=1e-06) for penb in self.pens_fitted])

    def fit_path(self,pen_list: List[float]):
        def register(Ue,Uold):
            # aligns columns of Ue towards those of Uold
            signs = np.sign(np.sum(Ue*Uold, axis=0))
            return Ue * signs

        for pen in pen_list:
            print(pen) # helpful to show progress # TODO: implement prettier progress bar
            first_fit = True
            if self._already_fitted(pen):
                print(f'already fitted pen {pen} so skipping estimation')
            else:
                print(f'attempting fit with pen {pen}') #for debugging
                Ue,Ve,te = self.data.fit_algo(self.algo,pen,self.K)
                if not first_fit:
                    Ue = register(Ue,Uold)
                    Ve = register(Ve,Vold)
                Uold,Vold = Ue,Ve
                first_fit = False
                self.save_ests_n_meta(Ue,Ve,te,pen)
                self.pens_fitted.add(pen)

    def save_ests_n_meta(self,Ue,Ve,te,pens):
        """
        Ue,Ve are pxK, qxK resp, single n,rs; can have different penalties in list pens,
        Warning: I've removed the cv column in meta!!!
        """
        d = self.data  # will access many attributes so convenient to abbreviate
        folder_name = d.folder
        os.makedirs(folder_name,exist_ok=True)
        file_name = folder_name+f'/estimates/K{self.K}_{self.algo}_W.csv'
        ests_to_csv(Ue,Ve,file_name)

        K = Ue.shape[1]
        keys = ['p','q','n','pen','K','t']
        vals = [d.p,d.q,d.n,pens,range(self.K),te]
        meta_df = pd.DataFrame(dict(zip(keys,vals)))
        if type(d) == MVNData: meta_df['rs'] = d.rs
        file_name = folder_name+f'/estimates/K{self.K}_{self.algo}_meta.csv'
        # want to be able to add more estimates if poor penalty parameters first
        # so if folder already exists then want to append without header
        headerq = not os.path.exists(file_name)
        meta_df.to_csv(file_name, mode='a',header = headerq)
        return print(f'saved to {file_name}')

    def load_ests_n_meta(self):
        folder_name = self.data.folder

        file_name = folder_name+f'/K{self.K}_{self.algo}_W.csv'
        stored_ests = np.array(pd.read_csv(file_name,header=None))

        file_name = folder_name+f'/K{self.K}_{self.algo}_meta.csv'
        meta = pd.read_csv(file_name,index_col=0)

        return stored_ests,meta

    def load_est_n_t(self,pen,nearest=False):
        K = self.K
        stored_ests,meta = self.load_ests_n_meta()
        if nearest == True:
            # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            potential_pens = set(meta['pen'])
            pen = min(potential_pens, key=lambda x:abs(x-pen))
        row_inds = np.isclose(meta['pen'],pen)
        assert np.all(meta[row_inds]['K'] == range(K)), 'wrong number of pairs'

        p = self.data.p
        We = (stored_ests[row_inds,:]).T
        Ue,Ve = We[:p,:],We[p:,:]
        # note: nonzero returns a tuple (of arrays), containing an array for each dimension/index
        idx1 = np.nonzero(row_inds)[0][0]
        te = meta.iloc[idx1]['t']
        return Ue,Ve,te

    def load_vrts(self,pen,inds,nearest=True):
        Ue,Ve,te = self.load_est_n_t(pen,nearest=nearest)
        d = self.data
        return d.X@Ue[:,inds], d.Y@Ve[:,inds]
    

class Split():
    """Very simple class just a struct really"""
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data

class CV():
    def __init__(self,data,folds,algo,K):
        self.data = data
        self.folds = folds
        self.K = K
        self.algo = algo

        self.splits = dict()
        self.solps = dict()
        self._get_splits()

        if type(self.data)==MVNData:
            self.full_path = MVNSolPath(data,algo,K)
        else:
            self.full_path = SolutionPath(data,algo,K)

    def _get_splits(self):
        kf = KFold(n_splits=self.folds,shuffle=True,random_state=0)
        X,Y = self.data.X,self.data.Y
        for fold,split in enumerate(kf.split(X,Y)):
            train_ids,test_ids = split
            folder_name = self.data.folder + f'/cv{fold}of{self.folds}'
            X_labs = self.data.X_labs
            Y_labs = self.data.Y_labs
            train_data = Data(X[train_ids],Y[train_ids],folder_name=folder_name,
                            X_labs=X_labs,Y_labs=Y_labs)
            test_data  = Data(X[test_ids],Y[test_ids],
                            X_labs=X_labs,Y_labs=Y_labs)
            self.splits[fold] = Split(train_data,test_data)
            # Added 30 Jan for biplot robustness...
            self.solps[fold] = SolutionPath(train_data,self.algo,self.K)

    def fit_algo(self,pen_list,full_also=True):
        if full_also:
            self.full_path.fit_path(pen_list)

        for split in self.splits.values():
            soln_path = SolutionPath(split.train_data,self.algo,self.K)
            soln_path.fit_path(pen_list)


    def process_cv(self,update=True):
        # update = True is addition on 28 April 2023 that will allow new penalties to be added without re-fitting previous penalties
        # key implementation detail is to find out which penalties have already been analysed
        
        # for use in multiple places later in this function
        if update:
            # load exisiting dfcv and create function to check if pen has already been processed
            # but if not run before, file won't exist so function should return False
            try:
                dfcv_full_old = self.load_dfcv()
                pens_processed = dfcv_full_old['pen'].unique()
                def already_processed(pen): return np.isclose(pen,pens_processed).any()
            except FileNotFoundError:
                print('FileNotFound being excepted')
                update = False 
                def already_processed(pen): return False
                
        dfcv_list = list()
        for fold,split in self.splits.items():
            train_data = split.train_data
            train_solp = SolutionPath(train_data,self.algo,self.K)
            test_data = split.test_data
            X_test,Y_test = test_data.X,test_data.Y

            def get_cv_df(pen):
                def analyse_cv(Ue,Ve,X_test,Y_test):
                    m,r = Ue.shape[:2]
                    cor_mat = utils.emp_cor_mat(Ue,Ve,X_test,Y_test)
                    cov_mat = utils.emp_cov_mat(Ue,Ve,X_test,Y_test)
                    diags = np.diag(cor_mat)
                    subsp = utils.can_corr_subs(cov_mat,r)
                    return np.block([diags,subsp])

                Ue,Ve,te = train_solp.load_est_n_t(pen)
                row_out = analyse_cv(Ue,Ve,X_test,Y_test)
                cols=(
                list(map(lambda r: 'rho'+str(r+1),range(self.K)))+
                list(map(lambda r: 'sub_corr'+str(r+1),range(self.K)))
                )
                my_dict = {k:[v] for k,v in zip(cols,row_out)}
                df = pd.DataFrame(my_dict)
                cv=f'{fold}of{self.folds}'
                df['cv'] = cv
                df['pen'] = pen
                for r in range(1,self.K+1):
                    inds = range(1,r+1)
                    df[f'rhosum{r}'] = sum([df[f'rho{i}'] for i in inds])
                return df

            # only process new penalties
            if update:
                pens_to_process = [pen for pen in train_solp.pens_fitted if not already_processed(pen)]
            else:
                pens_to_process = train_solp.pens_fitted
            print(update,pens_to_process)
            # pd.concat needs to have non-zero length list to work; and if already processed all penalties this will be the case
            if len(pens_to_process)>0:
                dfcv = pd.concat([get_cv_df(pen) for pen in pens_to_process])
                dfcv_list.append(dfcv)

        # again pd.concat needs non-zero length list to work so if no new penalties for any fold need empty dataframe for later concat
        dfcv_full = pd.concat(dfcv_list) if len(dfcv_list)>0 else pd.DataFrame()
        if update:
            dfcv_full = pd.concat([dfcv_full_old,dfcv_full])

        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfcv.csv'
        dfcv_full.to_csv(file_name,index=False)

        dfcv_avgd = (dfcv_full.drop(columns=['cv'])
                             .groupby(['pen']).agg([np.mean,np.std]))
        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfcvav.csv'
        dfcv_avgd.to_csv(file_name)

    def get_pen_list(self):
        solpaths = [SolutionPath(split.train_data,self.algo,self.K) for split in self.splits.values()]
        pens1 = list(solpaths[0].pens_fitted)
        def in_all_paths(pen):
            return np.all([solp._already_fitted(pen) for solp in solpaths])
        return [pen for pen in pens1 if in_all_paths(pen)]

    def process_stab(self,new_pens_only=True):
        # only compute stability for pens which haven't been processed yet
        # this assumes that we have the same stability objectives as previously
        # may want to change if decide to use different stability objectives
        if new_pens_only:
            # if no file found then need to update all penalties
            try:
                dfstabfull_old = self.load_dfstab()
                pens_processed = dfstabfull_old['pen'].unique()
                def already_processed(pen): return np.isclose(pen,pens_processed).any()
                pens_to_process = [pen for pen in self.get_pen_list() if not already_processed(pen)]
            except FileNotFoundError:
                new_pens_only = False
                pens_to_process = self.get_pen_list()
        else:
            pens_to_process = self.get_pen_list()
            
        folds = self.folds
        fold_pairs = [(i,j) for i in range(folds) for j in range(folds) if i<j]

        def get_sol(fold,pen):
            train_data = self.splits[fold].train_data
            train_solp = SolutionPath(train_data,self.algo,self.K)
            Ue,Ve,_ = train_solp.load_est_n_t(pen)
            return Solution(train_data,Ue,Ve)

        def stab_objs(sol1,sol2):
            """For Solution objects sol1,sol2, compute certain stability objectives 
            (see 'dct=...' in code for objectives; think these are pretty comprehensive)
            (later will think about custom objectives if needed)
            """
            # will compute weight and variate multiple sin-theta distances
            U1,V1 = sol1.Ue,sol1.Ve
            U2,V2 = sol2.Ue,sol2.Ve
            K = U1.shape[1]
            # can access full dataset via the CV object's self.data attribute
            X,Y = self.data.X,self.data.Y
            dct = {'wt_U': utils.sin_theta_mult(U1,U2),
                   'wt_V': utils.sin_theta_mult(V1,V2),
                   'vt_U': utils.sin_theta_mult(X @ U1, X @ U2),
                   'vt_V': utils.sin_theta_mult(Y @ V1, Y @ V2),}
            dct = OrderedDict(dct)
            # copied from process_estimates
            def column_labels(label):
                return list(map(lambda k: label+str(k+1),range(K)))
            # now make the dataframe
            row_out = np.block([dct[key] for key in dct])
            columns = np.concatenate([column_labels(key) for key in dct])
            df = pd.DataFrame(row_out.reshape(1,-1),columns=columns)
            return df
        
        def stabs(fold_pair,pen):
            """Return single row dataframe with stability scores for each objective"""
            sol1 = get_sol(fold_pair[0],pen)
            sol2 = get_sol(fold_pair[1],pen)
            df_row = stab_objs(sol1,sol2)
            df_row['pen'] = pen
            df_row['fold_pair'] = str(fold_pair)
            return df_row
        
        # potential for easy parallelization, but for now will do with base python
        df_list = [stabs(fold_pair,pen) for pen in pens_to_process for fold_pair in fold_pairs]
        df_full = pd.concat(df_list)

        if new_pens_only:
            df_full = pd.concat([dfstabfull_old,df_full])

        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfstabfull.csv'
        df_full.to_csv(file_name)


        dfstabav = (df_full.drop(columns=['fold_pair'])
                           .groupby(['pen']).agg([np.mean,np.std]))
        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfstabav.csv'
        dfstabav.to_csv(file_name)
        return dfstabav

    def load_dfstab(self):
        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfstabfull.csv'
        df = pd.read_csv(file_name,header=[0,],index_col=0)
        return df

    def load_dfstabav(self):
        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfstabav.csv'
        df = pd.read_csv(file_name,header=[0,1],index_col=0)
        return df

    def load_dfcv(self):
        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfcv.csv'
        df = pd.read_csv(file_name)
        return df

    def load_dfcvav(self):
        file_name = self.data.folder + f'/K{self.K}_{self.algo}_dfcvav.csv'
        df = pd.read_csv(file_name,header=[0,1],index_col=0)
        return df

    def get_best_pen(self,objective='rhosum5'):
        # be careful to call this with an objective that has actually been fitted already!
        df = self.load_dfcvav()
        df_sorted = df[objective].sort_values(by='mean',ascending=False)
        best_pen = df_sorted.index[0]
        return best_pen
    
    def get_cor_mat(self,split_no,pen):
        split = self.splits[split_no]
        test_data = split.test_data
        X_test,Y_test = test_data.X,test_data.Y

        train_data = split.train_data
        train_solp = SolutionPath(train_data,self.algo,self.K)
        Ue,Ve,te = train_solp.load_est_n_t(pen=pen,nearest=True)

        cor_mat = utils.emp_cor_mat(Ue,Ve,X_test,Y_test)
        return cor_mat
    

class Solution():
    """Very simple class just a struct really"""
    def __init__(self,data,Ue,Ve):
        self.data = data
        self.Ue = Ue
        self.Ve = Ve
        self.tup = (self.Ue,self.Ve,self.data.X,self.data.Y)
        #print(f'in sol class Ue[0,0] {Ue[0,0]}')


class MVNCV(CV):
    def __init__(self,data,folds,algo,K):
        super().__init__(data,folds,algo,K)
        self.full_path = MVNSolPath(data,algo,K)

    def fit_full(self,pen_list):
        self.full_path.fit_path(pen_list)

    def process_full(self):
        self.full_path.process_estimates()

    def load_dffull(self):
        return self.full_path.load_dffull()



# Now for special MVN classes...
class MVNSolPath(SolutionPath):
    def __init__(self,data,algo,K):
        super().__init__(data,algo,K)
        assert type(data) == MVNData, 'data has to be multivariate normal'

    def process_estimate(self,pen):
        # coding goal: rewrite this function to be more readable
        # will have more intuitive pairing of string labels to values

        # first compute all auxilliary quantities
        X,Y = self.data.X, self.data.Y
        Sig = self.data.mvn.Sig
        Ue,Ve,te = self.load_est_n_t(pen)
        p,K = Ue.shape[:2]
        U,V,D = utils.ccs_from_covariance(Sig,p)
        true_cor_mat = utils.true_cor_mat(Ue,Ve,Sig)

        # next match strings to values:
        dct = {'rho': np.diag(true_cor_mat),
               'R2s': utils.oracle_cos2thetas(Ue,Ve,Sig),
               'wt_u': utils.sin_theta_mult(Ue,U[:,:K],mode='successive'),
               'wt_v': utils.sin_theta_mult(Ve,V[:,:K],mode='successive'),
               'vt_u': utils.sin_theta_mult(Ue,U[:,:K],mode='successive',rel=Sig[:p,:p]),
               'vt_v': utils.sin_theta_mult(Ve,V[:,:K],mode='successive',rel=Sig[p:,p:]),
               'ld_u': utils.sin_theta_mult(Ue,U[:,:K],mode='successive',rel=Sig[:p,:p],sqrt=False),
               'ld_v': utils.sin_theta_mult(Ve,V[:,:K],mode='successive',rel=Sig[p:,p:],sqrt=False),
               'wt_U': utils.sin_theta_mult(Ue,U[:,:K]),
               'wt_V': utils.sin_theta_mult(Ve,V[:,:K]),
               'vt_U': utils.sin_theta_mult(Ue,U[:,:K],rel=Sig[:p,:p]),
               'vt_V': utils.sin_theta_mult(Ve,V[:,:K],rel=Sig[p:,p:]),
               'ld_U': utils.sin_theta_mult(Ue,U[:,:K],rel=Sig[:p,:p],sqrt=False),
               'ld_V': utils.sin_theta_mult(Ve,V[:,:K],rel=Sig[p:,p:],sqrt=False),
               }
        dct = OrderedDict(dct)

        def column_labels(label):
            return list(map(lambda r: label+str(r+1),range(K)))
        
        # now make the dataframe
        row_out = np.block([dct[key] for key in dct])
        columns = np.concatenate([column_labels(key) for key in dct])
        df = pd.DataFrame(row_out.reshape(1,-1),columns=columns)
        df['pen'] = pen
        df['t'] = te
        return df

    def process_estimates(self):
        single_rows = [self.process_estimate(pen) for pen in self.pens_fitted]
        dffull = pd.concat(single_rows)
        file_name = self.data.folder+f'/K{self.K}_{self.algo}_dffull.csv'
        dffull.to_csv(file_name,index=False)
        return dffull

    def fit_reprocess(self,pen_list):
        self.fit_path(pen_list)
        self.process_estimates()

    def load_dffull(self):
        file_name = self.data.folder+f'/K{self.K}_{self.algo}_dffull.csv'
        return pd.read_csv(file_name).sort_values(by='pen')



class MVNData(Data):
    def __init__(self,X,Y,rs,mvn_dist,folder_name):
        super().__init__(X,Y,folder_name)
        self.rs = rs
        self.mvn = mvn_dist
        self._check_dims_consistent()

    def _check_dims_consistent(self):
        # check units are consistent with parent distribution...
        msg = 'dims do not match parent distribution dims'
        assert (self.mvn.p == self.p) and (self.mvn.q == self.q), msg


class MVNDist():
    def __init__(self,p,q,Sig,cov_desc=None):
        self.p = p
        self.q = q
        self.Sig = Sig
        self._check_dims_consistent()

        self.cov_desc=cov_desc
        self.U,self.V,self.D = utils.ccs_from_covariance(Sig,p)

    def _check_dims_consistent(self):
        msg = 'Sig must be square with dims (p+q) x (p+q)'
        assert self.Sig.shape == (self.p+self.q,self.p+self.q),msg

    def gen_data(self,rs,n):
        X,Y = utils.data_from_covariance(self.Sig,self.p,self.q,n,rs)
        folder_name = interface.mvn_folder_name(self,rs,n)
        return MVNData(X,Y,rs,mvn_dist=self,folder_name=folder_name)

    def _get_folder_name(self,rs,n):
        # previously noted useful for metric_plotter #TODO check used there and delete if not
        return interface.mvn_folder_name(self,rs,n)


class MVNFactory():
    def __init__(self,cov_desc,machine):
        """
        Inputs:
        cov_desc - string summarising covariance type
        machine - function creating Sig from pair p,q
        """
        self.cov_desc = cov_desc
        self.machine = machine

    def build_mvn(self,p,q):
        Sig = self.machine(p,q)
        return MVNDist(p,q,Sig,cov_desc = self.cov_desc)