import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.scaffold.interface import algo_labels, get_cv_obj_from_data, get_cv_object, get_dataset
from src.scaffold.core import CV, MVNCV, MVNData
from src import utils

# global variables for nicer formatting
# allow larger font sizes for the labels and titles
label_size = 15
title_size = 20
label_padding = 15
title_pad = 10

# CORRELATION AND STABILITY ALONG TRAJECTORY
# ------------------------------------------
# There is a lot of repetition in the three different versions of these functions
# Would be nice to refactor to reduce this repetition (when time permits, TODO)

# Correlation - successive
def row_plot3(data,algos,nice_fn,inds,folder_title=False, legend_prefix='', y_label=''): #MVN=False):
    """Plot cv sums of nice_fn of test correlations for each algo in algos"""
    fig,axs = plt.subplots(ncols=len(algos),nrows=1,figsize=(4*len(algos)+4,5),sharey=True)
    df_dict = dict()
    for idx,algo in enumerate(algos):
        cv_obj = get_cv_obj_from_data(data,algo) #,MVN=MVN)
        ax,df = corr_plotter(cv_obj,nice_fn,inds,axs[idx], legend_prefix=legend_prefix)
        df_dict[algo] = df
    if folder_title: fig.suptitle(data.folder)
    # give first column the y-axis label
    axs[0].set_ylabel(y_label, fontsize=label_size, labelpad=label_padding)
    return fig, axs, df_dict

def corr_plotter(cv_obj,nice_fn,inds,ax,legend_prefix=''):
    logx = True if cv_obj.algo != 'ridge' else False
    ax,df = plot_cv(cv_obj,nice_fn,inds,logx=logx,ax=ax,legend_prefix=legend_prefix)
    ax.set_title(algo_labels[cv_obj.algo], fontsize=title_size, pad=title_pad)
    ax.grid()
    if cv_obj.algo!='wit':  ax.set_xlim([10**-4,10])
    if cv_obj.algo=='ridge': ax.set_xlim([0,1])
    return ax,df

def plot_cv(cv_object,nice_fn,inds,ax,logx=False,legend_prefix='sum_f_',):
    """
    Plot the sums of aggregation function applied to test correlations.
    
    Parameters:
    ----------
    cv_object: CV object
    
    nice_fn: scalar function
        This should be 'nice', in particular convex and such that f(x)>f(-x) for all x>0 (see overleaf)
    
    inds: list of ints
        indices up to which to aggregate
    """
    dfcvfull = cv_object.load_dfcv()

    # see if cv_object is from synthetic MultiVariate Normal, if so will plot true values...
    MVN = (type(cv_object)==MVNCV)

    # extract relevant information from dfcvfull
    # note inds is zero indexed, while rhos are 1 indexed so need to convert
    # (this gave me a nasty bug for a while, careful if modifying!)
    rhos = ['rho'+str(k) for k in range(1,cv_object.R)]
    cols_of_interest = ['pen']+rhos
    df = dfcvfull[cols_of_interest].copy().set_index('pen')
    
    # penalty is the index column so won't be transformed by nice_fn (which is good)
    # first apply f, then cumulative sum, and only then do the groupby
    df = df.apply(nice_fn)
    df.columns = [legend_prefix+str(k) for k in range(1,cv_object.R)]
    
    df_cumsum = df.apply(lambda x: x.cumsum(), axis=1)
    df_sum_f_av = df_cumsum.groupby(level=0).agg([np.mean,np.std])

    # get true values if available
    if MVN:
        df_true = cv_object.load_dffull()[cols_of_interest].set_index('pen')
        df_true_sum_f = df_true.apply(nice_fn).cumsum(axis=1)
        df_true_sum_f.columns = df.columns


    # finally plot, using sensible colour scheme
    cols_to_plot = df.columns[inds]
    cm = plt.get_cmap('winter')
    for idx,col in enumerate(cols_to_plot):
        frac = idx/(len(inds)) if len(inds)>1 else 0
        color = cm(frac)
        df_sum_f_av[col].plot(ax=ax,y='mean',yerr='std',logx=logx,label=col,color=color)
        if MVN:
            # change formatting to have small dots for markers, and dashed lines
            df_true_sum_f[col].plot(ax=ax,logx=logx,color=color,style='.-',linewidth=0.5,linestyle='--')

    return ax,df_sum_f_av

# Correlation - successive
def subsp_row_plot(data,algos,inds,folder_title=False): #MVN=False):
    """Plot cv sums of nice_fn of tesst correlations for each algo in algos"""
    fig,axs = plt.subplots(ncols=len(algos),nrows=1,figsize=(4*len(algos)+4,5),sharey=True)
    df_dict = dict()
    for idx,algo in enumerate(algos):
        cv_obj = get_cv_obj_from_data(data,algo) #,MVN=MVN)
        ax,df = subsp_corr_plotter(cv_obj,inds,axs[idx])
        df_dict[algo] = df
    if folder_title: fig.suptitle(data.folder)
    return fig, df_dict

def subsp_corr_plotter(cv_obj,inds,ax):
    logx = True if cv_obj.algo != 'ridge' else False
    ax,df = plot_cv_subspace(cv_obj,inds,logx=logx,ax=ax)
    ax.set_title(algo_labels[cv_obj.algo])
    ax.grid()
    if cv_obj.algo!='wit':  ax.set_xlim([10**-4,10])
    if cv_obj.algo=='ridge': ax.set_xlim([0,1])
    return ax,df

def plot_cv_subspace(cv_object,inds,ax,logx=False):
    """
    Plot subspace test correlations - whatever aggregation was used in process_cv call.
    """
    dfav = cv_object.load_dfcvav()

    sub_corrs = ['sub_corr'+str(k) for k in inds]
    #  plot, using sensible colour scheme
    cm = plt.get_cmap('winter')
    for idx,col in enumerate(sub_corrs):
        frac = idx/(len(inds)) if len(inds)>1 else 0
        color = cm(frac)
        dfav[col].plot(ax=ax,y='mean',yerr='std',logx=logx,label=col,color=color)

    return ax,dfav


# Stability
def stab_row_plot(data,algos,criteria,inds,folder_title=False):
    """Plot cv sums of the different instability criteria for each algo"""
    # squeeze=False ensures that axs is a 2D array
    fig, axs = plt.subplots(ncols=len(algos),nrows=len(criteria),figsize=(4*len(algos)+4,5*len(criteria)),sharey=True, squeeze=False)
    df_dict = dict()
    for idx,algo in enumerate(algos):
        left_col = True if idx==0 else False
        cv_obj = get_cv_obj_from_data(data,algo) #,MVN=MVN)
        for jdx,criterion in enumerate(criteria):
            top_row = True if jdx==0 else False
            # stab_plotter calls plot_instability and plots CV instability; when MVN it also plots oracle quantities
            ax,df = stab_plotter(cv_obj,criterion,inds,axs[jdx,idx],top_row=top_row)
            df_dict[(algo,criterion)] = df
            suff = 'k(-cv)' if isinstance(data,MVNData) else 'k-cv'
            if left_col: ax.set_ylabel(criterion+suff, fontsize=label_size, labelpad=label_padding)
            
            # sort out the legend
            # After all the plotting is done, get the lines and labels of the current axes
            lines, labels = ax.get_legend_handles_labels()   
            # If legend_display is 'compact', remove the lines and labels for dffull
            lines, labels = zip(*[(line, ind) for line, label, ind in zip(lines, labels, 2*inds) if '_cv' in label])
            # Set the legend with the desired lines and labels
            ax.legend(lines, labels,loc='upper right')

    if folder_title: fig.suptitle(data.folder)
    return fig, df_dict

def stab_plotter(cv_obj,criterion,inds,ax,top_row=True):
    logx = True if cv_obj.algo != 'ridge' else False
    ax,df = plot_instability(cv_obj,criterion,inds,ax=ax,logx=logx,)
    if top_row:
        ax.set_title(algo_labels[cv_obj.algo], fontsize=title_size, pad=title_pad)
    ax.grid()
    if cv_obj.algo!='wit':  ax.set_xlim([10**-4,10])
    if cv_obj.algo=='ridge': ax.set_xlim([0,1])
    return ax,df

def plot_instability(cv_object,criterion,inds,ax,logx=False,legend_display='compact'):
    """Plot specified metrics for given CV object.
    The CV object may be MVNCV or CV.

    Parameters:
    ----------
    cv_object: CV object

    criterion: {'wt_u','wt_U','vt_u','vt_U','wt_v','wt_V','vt_v','vt_V'}
        criterion to plot
    
    inds: list of ints
        indices to plot
    """
    dfstabav = cv_object.load_dfstabav()
    # see if cv_object is from synthetic MultiVariate Normal, if so will plot true values...
    MVN = (type(cv_object)==MVNCV)

    cols_to_plot = [criterion+str(k) for k in inds]
    
    # get true values if available
    if MVN:
        # sort values to prevent line wiggling
        dffull = cv_object.load_dffull().sort_values(by='pen').set_index('pen')
        
    cmap = plt.get_cmap('autumn')
    mycmap = lambda z: cmap(0.8*z) 

    for idx,(k,col) in enumerate(zip(inds,cols_to_plot)):
        frac = idx/(len(cols_to_plot)-1) if len(cols_to_plot)>1 else 0
        color = mycmap(frac)
        label = col+'_cv' #k if legend_display=='compact' else 
        dfstabav[col].plot(y='mean',yerr='std',logx=logx,
                            ax=ax,style=':',label=label,color=color)# ,color='purple')
    
        if MVN:
            label = col #None if legend_display == 'compact' else
            dffull[col].plot(ax=ax,logx=logx,color=color,style='.-',linewidth=0.5,linestyle='--',label=label)

    return ax,dfstabav[cols_to_plot]



# CORRELATION DECAY
# -----------------

def plot_corr_decr(cv_obj,pen,Kmax=8,nearest=True,ax=None,min_y=None,squared=False,alg_title=True):
    if nearest == True:
        # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
        potential_pens = cv_obj.get_pen_list()
        pen = min(potential_pens, key=lambda x:abs(x-pen))

    dfcv = cv_obj.load_dfcv()
    relev = dfcv[np.isclose(dfcv['pen'],pen)]
    rhos = ['rho'+str(n) for n in range(1,Kmax)]

    df = relev[['pen','cv']+rhos].copy()
    if squared: df[rhos] = df[rhos]**2
    df_points = (pd.wide_to_long(df,['rho'],i='cv',j='k').
                 reset_index())
    df_means = df_points.groupby('k').mean(numeric_only=True).reset_index()

    if ax is None: fig,ax = plt.subplots()
    sns.scatterplot(x='k',y='rho',hue='cv',data=df_points,ax=ax)
    sns.lineplot(x='k',y='rho',data=df_means,ax=ax,color='black',marker='X',markersize=10)
    
    if alg_title: title_start = algo_labels[cv_obj.algo] + ',  '
    else: alg_title = ''
    ax.set_title(title_start + 'pen ' + ('%s' % float('%.2g' % pen))) # 2 s.f.
    if min_y!=None: 
        ax.set_ylim([min_y,1])
        ax.axhline(y=0, color='black', linestyle='--')  # horizontal line at 0
    ax.grid()
    return ax

def triple_plot(dataset, algo, pens=None, min_y=None, squared=False, compact=True):
    cv_obj = get_cv_object(dataset,algo)
    if pens is None:
        df = cv_obj.load_dfcvav()
        top5 = np.array(df[('rhosum5','mean')].sort_values(ascending=False).index[:5]).round(5)
        best_pen = top5[0]
        pens = np.array([0.1,1,10])*best_pen
        return triple_plot(dataset,algo,pens,squared)
    else:
        # pens should be iterable...
        fig,axs = plt.subplots(ncols=len(pens),figsize=(15,5),sharey=True)
        for idx,pen in enumerate(pens):
            plot_corr_decr(cv_obj,pen=pen,Kmax=9,nearest=True,ax=axs[idx],min_y=min_y,squared=squared)
        if not compact: fig.suptitle(f'Dataset {dataset}, algo {algo}')
        return fig

def corr_decay_misc(dataset, pairs, min_y=None, squared=False,figsize=(15,5)):
    """Plot correlation decay for each (algorithm,penalty) pair in pairs
    
    Here pairs are (algorithm,penalty) Pairs
    Expected to have 3 Pairs
    If squared then plot squared correlations instead (which may appear to decay faster)
    """
    data = get_dataset(dataset)
    fig,axs = plt.subplots(ncols=len(pairs),figsize=figsize,sharey=True)        
    for idx,(algo,pen) in enumerate(pairs):
        cv_obj = get_cv_obj_from_data(data,algo)
        plot_corr_decr(cv_obj,pen=pen,ax=axs[idx],Kmax=9,nearest=True,min_y=min_y,squared=squared,alg_title=True)
        #axs[idx].set_title(f'{algo}, pen {pen}')
    #fig.suptitle(f'Dataset {dataset}')
    return fig
        

# correlation matrices for subspace overlap
def c2c2ns(cmat,plot=False,ax=None):
    """Convert correlations to squared correlations with row/columns sums.
    If plot=True, plot the result.
    Abbreviation derivation: Corr 2 Corr^2 aNd col/row Sums"""
    cmat22 = cmat**2
    row_sums = np.sum(cmat22,axis=1).reshape(-1,1)
    col_sums = np.sum(cmat22,axis=0)
    comb = np.block([[cmat22,row_sums],[col_sums,np.nan]])
    if plot:
        # all entries are squares so \gt 0, cut the center=0,
        fig = sns.heatmap(comb, annot=True,vmin=0,vmax=1,cmap='Blues',ax=ax)
        len_idx = len(col_sums)
        ax.vlines(len_idx, ymin=-1, ymax=len_idx+1, color="white", linewidth=5)
        ax.hlines(len_idx, xmin=-1, xmax=len_idx+1, color="white", linewidth=5)
        return fig
    else:
        return comb

def corr_mat_plot(cmats):
    """Plot heat map for each correlation matrix in cmats.
    cmats is dictionary of correlation matrices"""
    fig,axs = plt.subplots(ncols = len(cmats),figsize=(8*len(cmats)-2,6))
    for idx,key in enumerate(cmats):
        cmat = cmats[key]
        sns.heatmap(cmat,annot=True,vmin=-1,vmax=1,center=0,cmap='RdBu',ax=axs[idx])
    return fig, axs

def sign_expl_plot(cmats):
    """cmats is dictionary of correlation matrices"""
    fig,axs = plt.subplots(ncols = len(cmats),figsize=(8*len(cmats)-2,6))
    for idx,key in enumerate(cmats):
        cmat = cmats[key]
        c2c2ns(cmat,plot=True,ax=axs[idx])
    return fig, axs

def double_plot(cmats,suffix):
    """First plot correlation matrices, then plot squared correlations with row/column sums"""
    fig1, _ = corr_mat_plot(cmats)
    plt.suptitle("Correlations; " + suffix)
    fig2, _ = sign_expl_plot(cmats)
    plt.suptitle("Squared Corrs: " + suffix)
    return fig1, fig2

def get_corr_mats_along_path(dataset,algo,pens,inds,ref_pen,reg='signs'):
    cmats = dict()
    solp = get_cv_object(dataset,algo).full_path
    Zref = solp.load_vrts(pen=ref_pen,inds=inds)[0] # second component is for Zy i.e. V variates
    d = solp.data

    for pen in pens:
        Ue,Ve,_ = solp.load_est_n_t(pen,nearest=True)
        Zx = d.X @ Ue[:,inds]
        Zx = utils.register_general(Zx, Zref, reg_method = reg, output = 'transformed_Z')

        cmats[pen] = utils.col_corr(Zref,Zx)
    return cmats

def path_viz(dataset,algo,pens,inds=range(5),ref_pen=0.05,reg='signs'):
    """Apply double_plot to output from get_corr_mats_along_path."""
    cmats = get_corr_mats_along_path(dataset,algo,pens,inds=inds,ref_pen=ref_pen,reg=reg)
    pens_str = [f'{pen:.1e}' for pen in pens]
    label_str_suffix = f"Dataset: {dataset}, algo: {algo}, ref_pen: {f'{ref_pen:.1e}'}, pens: {', '.join(pens_str)}"
    return double_plot(cmats,label_str_suffix)

# Stability between folds
def get_corr_mats_folds(dataset,algo,ref_pen,inds,folds,reg='signs'):
    cmats = dict()
    cv_obj = get_cv_object(dataset,algo)
    fullp = cv_obj.full_path
    Zref = fullp.load_vrts(pen=ref_pen,inds=inds)[0] # second component is for Zy i.e. V variates
    d = fullp.data

    for fold in folds:
        solp = cv_obj.solps[fold]
        Ue,Ve,_ = solp.load_est_n_t(ref_pen,nearest=True)

        Zx = d.X @ Ue[:,inds]
        Zx = utils.register_general(Zx, Zref, reg_method = reg, output = 'transformed_Z')

        cmats[fold] = utils.col_corr(Zref,Zx)
    return cmats

def viz_folds(dataset,algo,folds,ref_pen=0.05,inds=range(5),reg='signs'):
    cmats = get_corr_mats_folds(dataset,algo,ref_pen,folds=folds,inds=inds,reg=reg)
    fold_str = ', '.join([str(fold) for fold in folds])
    label_str_suffix = f"Dataset: {dataset}, algo: {algo}, ref_pen: {f'{ref_pen:.1e}'}, folds: {fold_str}"
    return double_plot(cmats,label_str_suffix)


# Stability via correlation matrices for miscellaneous estimators
def get_corr_mats_misc(dataset, pairs, ref_pair_idx=1, inds=range(5),ref_vrt='Zx',reg='signs',fold='full'):
    """
    Warning: current version always uses Zx for horizontal, but vertical can be Zy
    This is not what one might expect.
    
    Here pairs are (algorithm,penalty) Pairs
    Expected to have 3 Pairs

    Registration options: None, 'signs', 'orthog', 'perms_n_signs'

    If fold='full', then we use full path, otherwise we use fold path and test data for evaluation
    """
    cmats=dict()
    data = get_dataset(dataset)

    # get evaluation data
    if fold=='full':
        X_eval,Y_eval = data.X, data.Y
    else:
        # use test data associated to specified fold
        algo = pairs[0][0] #arbitrary algo
        cv_object = get_cv_obj_from_data(data,algo)
        test_data = cv_object.splits[fold].test_data
        X_eval,Y_eval = test_data.X, test_data.Y

    # can't reuse cv_object because different algos
    def get_pen_path(pair):
        algo,pen = pair
        if fold=='full': return pen,get_cv_obj_from_data(data,algo).full_path
        else: 
            assert type(fold)==int, 'fold type must be integer'
            return pen,get_cv_obj_from_data(data,algo).solps[fold]

    # first get reference, then can define eval data
    pen_ref,path_ref = get_pen_path(pairs[ref_pair_idx])
    #vrt_index = 0 if ref_vrt=='Zx' else 1
    
    # need to change load_vrts to use X_eval.
    # !!!!!!!! 
    # Next thing to do after lunch 
    Ue,Ve,_ = path_ref.load_est_n_t(pen_ref,nearest=True)
    Zref = X_eval @ Ue[:,inds] if ref_vrt=='Zx' else Y_eval @ Ve[:,inds]
    #Zref = path_ref.load_vrts(pen=pen_ref,inds=inds)[vrt_index]

    # now evaluate
    for idx,pair in enumerate(pairs):
        pen,fullp = get_pen_path(pair)
        Ue,Ve,_ = fullp.load_est_n_t(pen,nearest=True)
        Zx = X_eval @ Ue[:,inds]
        Zx = utils.register_general(Zx, Zref, reg_method = reg, output = 'transformed_Z')
        cmats[idx] = utils.col_corr(Zref,Zx)

    return cmats


def viz_mats_misc(dataset,pairs, ref_pair_idx=1, inds=range(5),ref_vrt='Zx',reg='signs',fold='full'):
    cmats = get_corr_mats_misc(dataset, pairs, ref_pair_idx, inds, ref_vrt,reg,fold=fold)
    # convert pen part of pairs from float to string in scientific notation 1 decimal place, keeping as (algo,pen) format
    pairs_str = ', '.join(['('+algo_labels[algo]+', '+f'{pen:.1e}'+')' for algo,pen in pairs])
    label_str_suffix = f"Dataset: {dataset}, pairs: {pairs_str}, ref_idx: {ref_pair_idx}"
    return double_plot(cmats,label_str_suffix)

def sq_overlap_misc(dataset, pairs, ref_pair_idx=1, inds=range(5), ref_vrt='Zx', reg='signs', fold='full'):
    cmats = get_corr_mats_misc(dataset, pairs, ref_pair_idx, inds, ref_vrt,reg,fold=fold)
    fig, axs = sign_expl_plot(cmats)
    for idx, (ax, (algo, pen)) in enumerate(zip(axs, pairs)):
        title = algo_labels[algo] + ',  pen ' + ('%s' % float('%.2g' % pen)) # 2 s.f.
        ax.set_title(title, fontsize=label_size, pad=title_pad)
        if idx == ref_pair_idx:
            axs[0].set_ylabel(title, fontsize=label_size, labelpad=label_padding)
    return fig, axs

# edited to have more informative row and column labels
# would be nice to add in option of overall title with extra argument information, with compact flag
def sq_overlap_folds(dataset,algo,folds,ref_pen=0.05,inds=range(5),reg='signs'):
    cmats = get_corr_mats_folds(dataset,algo,ref_pen,folds=folds,inds=inds,reg=reg)
    fig, axs = sign_expl_plot(cmats)
    for ax, fold in zip(axs, folds):
        title = 'fold ' + str(fold)
        ax.set_title(title, fontsize=label_size, pad=label_padding)
    axs[0].set_ylabel('full sample', fontsize=label_size, labelpad=label_padding)
    return fig, axs


def sq_overlap_path(dataset,algo,pens,inds=range(5),ref_pen=0.05,reg='signs'):
    cmats = get_corr_mats_along_path(dataset,algo,pens,inds=inds,ref_pen=ref_pen,reg=reg)
    fig, axs = sign_expl_plot(cmats)
    for ax, pen in zip(axs, pens):
        title = 'pen ' + ('%s' % float('%.2g' % pen))
        ax.set_title(title, fontsize=label_size, pad=label_padding)
    axs[0].set_ylabel('pen ' + ('%s' % float('%.2g' % ref_pen)), fontsize=label_size, labelpad=label_padding)
    return fig, axs

# array of trajectory similarities...
# rewrite to take dict
def algo_penlist_pairs_to_algo_pen_pairs(algo_penlist_pairs):
    def get_pairs(algo_penlist_pair):
        algo = algo_penlist_pair[0]
        pens = algo_penlist_pair[1]
        return [(algo,pen) for pen in pens]
    # get list of lists (lol) of pairs
    pairs_lol = [get_pairs(algo_penlist_pair) for algo_penlist_pair in algo_penlist_pairs]
    # now turn list of lists into single list
    pairs = [pair for sublist in pairs_lol for pair in sublist]
    return pairs


def traj_stab_array(dataset,algo_penlist_pairs,inds,space='variate'):
    """
    Parameters
    ----------
    dataset : str
        Name of dataset, to be passed to get_dataset
    
    algo_penlist_pairs : list of pairs of form (algo,penlist) where algo is string for algorithm and penlist is 1D numpy array 
        This defines the estimators to be plotted
        (original implementation idea was OrderedDict, but thought tuple would be simpler)
        Note the lists of penalties can be of different lengths

    inds : list of ints
        Indices of variates to use

    space : {'variate','direction'}
        In which space to work in for squared sin-theta-distances

    Returns
    -------
    array of shape (d,d,|inds|) where d is sum of length of penalty lists in algo_penlist_pairs
    """
    def discrepancy(pair1,pair2):
        """
        Takes two (algo,pen) pairs and returns vector of sin_theta distances between estimates for given indices
        (admittedly bit strange to do sin_theta_mult on arbitrary inds, but will leave for now)
        """
        data = get_dataset(dataset)
        X = data.X

        def get_Me(pair):
            """Matrices from estimates to compare with sin-theta-mult, either variates or directions"""
            algo = pair[0]
            pen = pair[1]
            solp = get_cv_object(dataset,algo).full_path
            Ue,_,_ = solp.load_est_n_t(pen,nearest=True)
            if space=='variate': return X @ Ue[:,inds]
            elif space=='direction': return Ue[:,inds]
            else: raise ValueError(f"space must be 'variate' or 'direction', not {space}")

        Me1,Me2 = map(get_Me,[pair1,pair2])

        return utils.sin_theta_mult(Me1,Me2)


    pairs = algo_penlist_pairs_to_algo_pen_pairs(algo_penlist_pairs)
    I = len(pairs)
    dist_inds = [(i,j) for i in range(I) for j in range(I) if i<j]
    discrs = map(lambda pij: discrepancy(pairs[pij[0]],pairs[pij[1]]), dist_inds)

    arr = np.zeros((I,I,len(inds)))

    for idx_ps,out in zip(dist_inds,discrs):
        arr[idx_ps[0],idx_ps[1],:] = arr[idx_ps[1],idx_ps[0],:] = out

    # we generally prefer sin-2 theta to sin-theta for comparison between subsequent directions
    return arr


from itertools import product
from collections import OrderedDict

def pretty_arr_plot(arr,algo_penlist_pairs,ax=None):
    """
    Parameters
    ----------
    arr : array of shape (d,d) where d is sum of length of penalty vectors for pairs in algo_penlist_pairs
    
    algo_penlist_pairs : list of pairs of form (algo,penlist) where algo is str and penlist is 1D numpy array

    ax :  matplotlib axis object, optional
    """
    # next up, change this to target syntax from above...


    # 0. get axis if needed
    newfig = False
    if ax is None: 
        newfig = True
        fig, ax = plt.subplots(figsize=(12, 9)); print('done')

    # 1. put relevant information as friendly multi-index in dataframe
    pairs = algo_penlist_pairs_to_algo_pen_pairs(algo_penlist_pairs)
    # convert algo names to algo labels
    idx = pd.MultiIndex.from_tuples(pairs,names=['algo','pen'])
    df = pd.DataFrame(arr,columns=idx,index=idx)
    sns.heatmap(df, annot=True, ax=ax)
    # 2. adapt answer from stackoverflow to print the dataframe
    # https://stackoverflow.com/questions/64234474/how-to-customize-y-labels-in-seaborn-heatmap-when-i-use-a-multi-index-dataframe
    label_mapping = OrderedDict()
    for algo, pen in df.index:
        label_mapping.setdefault(algo, [])
        # convert pen to string with scientific notation and one decimal places
        pen = "{:.1e}".format(pen)
        label_mapping[algo].append(pen)

    line_breaks = []
    new_labels = []
    for algo, pen_list in label_mapping.items():
        pen_list[0] = "{} - {}".format(algo_labels[algo], pen_list[0])
        new_labels.extend(pen_list)

        if line_breaks:
            line_breaks.append(len(pen_list) + line_breaks[-1])
        else:
            line_breaks.append(len(pen_list))

    len_idx = len(idx)
    ax.vlines(line_breaks, ymin=-1, ymax=len_idx, color="white", linewidth=5)
    ax.hlines(line_breaks, xmin=-1, xmax=len_idx, color="white", linewidth=5)
    ax.set_yticklabels(new_labels)
    ax.set_xticklabels(new_labels)

    if newfig: return fig,ax

def plot_array(arr,algo_penlist_pairs):
    """
    Parameters
    ----------
    arr : array of shape (d,d,k) where d is sum of length of penalties for pairs in targets and k is number of indices in inds used to generate it with traj_stab_array
    algo_penlist_pairs : list of pairs of form (algo,penlist) where algo is str and penlist is 1D numpy array

    Idea
    ----
    Plot arr[:,:,k] for each k with pretty_arr_plot
    """
    for k in range(arr.shape[2]):
        ax = pretty_arr_plot(arr[:,:,k],algo_penlist_pairs)
        ax.set_title('sin_theta_k variate, X view with k = {}'.format(k))
        plt.show()
    
def find_smallest_submatrix(matr,ind_partition, method='brute_force'):
    """
    Parameters
    ----------
    matr : square numpy array
        Matrix to find submatrix of

    ind_partition : list of lists of ints
        Each sublist is a partition of the indices of matr
        (so each sublist is a subset of {0,...,len(matr)-1})

    method : str, optional
        Method to use to find the best submatrix
        Currently only 'brute_force' is implemented
    
    Returns
    -------
    best_inds : list of ints
        Indices of matr corresponding to best submatrix, i.e. submatrix with the smallest entrywise infinity norm (i.e. absolute value of the largest element)
        This must contain exactly one index from each of the ind_partitions
    """
    if method=='brute_force':
        best_inds = None
        best_val = np.inf
        for inds in product(*ind_partition):
            submatr = matr[np.ix_(inds,inds)]
            val = np.max(np.abs(submatr))
            if val < best_val:
                best_val = val
                best_inds = inds
        return best_inds
    
def find_smallest_submatrix_from_penlist_pairs(matr,algo_penlist_pairs, method='brute_force'):
    """
    Parameters
    ----------
    matr : square numpy array
        Matrix to find submatrix of

    algo_penlist_pairs : list of pairs of form (algo,penlist) where algo is str and penlist is 1D numpy array
        Each pair corresponds to a row/column of matr, and the submatrix returned will contain exactly one row/column from each pair (i.e. from each algorithm)

    method : str, optional
        Method to use to find the best submatrix
        Currently only 'brute_force' is implemented
    
    Returns
    -------
    best_inds : list of ints
        Indices of matr corresponding to best submatrix, i.e. submatrix with the smallest entrywise infinity norm (i.e. absolute value of the largest element)
        This must contain exactly one index from each of the ind_partitions
    """
    # 1. get indices corresponding to algo_penlist_pairs
    pairs = algo_penlist_pairs_to_algo_pen_pairs(algo_penlist_pairs)
    #2. get corresponding index partition from these pairs (i.e. list of lists of indices for each algo), the penalties are not distinct so we select on algorithm
    ind_partition = [np.where(np.array(pairs)[:,0]==algo)[0] for algo,penlist in algo_penlist_pairs]
    # 3. get indices of best submatrix    
    best_inds = find_smallest_submatrix(matr,ind_partition,method=method)
    
    #4. get best pairs, and best submatrix, then package into a pandas dataframe with these best_pairs as index and columns
    best_pairs = [pairs[i] for i in best_inds]
    best_submatr = matr[np.ix_(best_inds,best_inds)]
    # get multiindex from best_pairs
    ##pair_index = pd.MultiIndex.from_tuples(best_pairs,names=['algo','pen'])
    # convert pen column in this multi-index to string with scientific notation and one decimal place
    # avoid the FutureWarning: inplace is deprecated and will be removed in a future version.
    #pair_index.set_levels([pair_index.levels[0],["{:.1e}".format(pen) for pen in pair_index.levels[1]]],inplace=True)
    pair_index = pd.MultiIndex.from_tuples([(algo,"{:.1e}".format(pen)) for algo,pen in best_pairs],names=['algo','pen'])
    df = pd.DataFrame(best_submatr,columns=pair_index,index=pair_index)

    return best_pairs, best_inds, best_submatr, df


# Heap plots for sorted squared structure correlations
def sorted_sc2_caller(dataset,algo,pen,inds=[0,1],vrt='Zx',vbl='X',cumulative=False):
    """Return array of squared structure correlations in decreasing order
    Calls sorted_sc2"""
    solp = get_cv_object(dataset,algo).full_path
    return sorted_sc2(solp,inds,pen,vrt=vrt,vbl=vbl,cumulative=cumulative)

def sorted_sc2(solp,inds,pen,vrt='Zx',vbl='X',cumulative=False):
    """Return array of squared structure correlations in decreasing order"""
    Ue, Ve, te = solp.load_est_n_t(pen,nearest=True)
    d = solp.data

    vrts = d.X @ Ue[:,inds] if vrt=='Zx' else d.Y @ Ve[:,inds]
    vbls = d.X if vbl=='X' else d.Y

    corrs = utils.col_corr(vbls,vrts)

    # Plot the largest squared structure correlations in decreasing order
    if not cumulative:
        return np.sort((corrs**2).sum(axis=1))[::-1]
    else:
        # first square the correlations then get cumulative sums of variates
        sq_corrs = corrs**2
        cumsums = np.cumsum(sq_corrs,axis=1)
        # now sort each column separately into decreasing order
        return np.sort(cumsums,axis=0)[::-1]


def heap_comparison(dataset,algo_pen_pairs,inds,vrt='Zx',vbl='X',scale=5,cumulative=False):
    """Plot heap plots for each algo-pen pair in a row, returning fig,axes.
    
    Parameters
    ----------
    dataset : {'nutrimouse','microbiome','breastdata'}
        string representing dataset
    algo_pen_pairs : list of pairs of form (algo,pen) where algo is str and pen is float
        Each pair will give a subplot in the figure
    inds : list of ints
        Indices of canonical variates to use (and sum squared correlations over)
    vrt : {'Zx','Zy'}, optional
        Whether to use Zx or Zy for the canonical variates
    vbl : {'X','Y'}, optional
        Whether to use X or Y for the variables to correlate with the canonical variates
    scale : float, optional
        Scale factor for the figure size
    cumulative : bool, optional
        Whether to plot cumulative heap plots (i.e. cumulative sum of squared correlations)
    """
    n = len(algo_pen_pairs)
    fig, axes = plt.subplots(1,n,figsize=(scale*n,scale),sharey=True)
    for i,(algo,pen) in enumerate(algo_pen_pairs):
        sc2 = sorted_sc2_caller(dataset,algo,pen,inds,vrt=vrt,vbl=vbl,cumulative=cumulative)
        if not cumulative:
            axes[i].plot(sc2)
            axes[i].set_title(f"{algo_labels[algo]}, {pen:.1e}")
        else:
            # for each algorithm plot each cumulative sum in the same subplot
            # want gradual colour gradient
            cm = plt.get_cmap('winter')
            # better heatmap that has bigger change of color
            cm = plt.get_cmap('seismic')
            for j in range(sc2.shape[1]):
                color = cm(1.*j/sc2.shape[1])
                axes[i].plot(sc2[:,j],label=f"{j+1}",color=color)
            axes[i].set_title(f"{algo_labels[algo]}, {pen:.1e}")
            axes[i].legend()
    return fig, axes