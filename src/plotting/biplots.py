# Biplot visualisations using Plotly

# Comments:
# These evolved organically after looking at a number of biplots
# I did some mild refactoring, to remove most repetition, but there is scope to restructure to increase modularity

# external imports
import numpy as np 
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# internal imports
from src.algos import algo_labels
from src.scaffold.wrappers import get_dataset, get_cv_obj_from_data
from src import utils

def gen_circle():
    circle=pd.DataFrame()
    circle['theta'] = np.linspace(-np.pi,np.pi,201)
    circle['x'] = np.cos(circle['theta'])
    circle['y'] = np.sin(circle['theta'])
    return circle

def gen_diamond():
    # to test: gen_diamond().plot(x='x',y='y')
    top_half = pd.DataFrame()
    top_half['x'] = np.linspace(-1,1,101)
    top_half['y'] = np.abs(1-np.abs(top_half['x']))
    return pd.concat([top_half,-top_half],axis=0)

def my_sphere(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    # fixing colour requires some fiddling
    # this stackoverflow was useful: https://stackoverflow.com/questions/53992606/plotly-different-color-surfaces
    # 0.2 on greys colorscale was about right
    sc = 0.5 * np.ones(len(u))
    # this didn't end up working on my macbook so I just switched colorscale from Greys to gray
    # better on PC with Greys
    # use whichever gives better effect
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=sc, opacity=0.7,colorscale='Greys',cmin=0,cmax=1,cauto=False,showscale=False)


# Converting dictionary of style functions to dictionary of style values
def apply_style_functions(labs,function_dictionary,colorscale='Bluered_r'):
        """ Apply style functions to labels to return dictionary of marker styles.
        function_dictionary should have keys 'color','symbol','cmin','cmax'"""
        # relevant doc page is https://plotly.com/python/reference/scatter/#scatter-marker
        out_dict = dict()
        if 'color' in function_dictionary:
            if type(function_dictionary['color']) is str:
                out_dict['color'] = function_dictionary['color']
            else:
                out_dict['color'] = function_dictionary['color'](labs)
                if 'cmin' in function_dictionary:
                    out_dict['colorscale'] = colorscale
                    out_dict['cmin'] = function_dictionary['cmin']
                    out_dict['cmax'] = function_dictionary['cmax']
        if 'size' in function_dictionary:
            if type(function_dictionary['size']) in [str,int]:
                out_dict['size'] = function_dictionary['size']
            else:
                # function_dictionary['size'] should be a function that takes labs as input
                out_dict['size'] = function_dictionary['size'](labs)
        if 'symbol' in function_dictionary:
            out_dict['symbol'] = function_dictionary['symbol'](labs)
        return out_dict
    
# The main function used
def get_trs_load_n_weights(solp,pen,inds,Zref,masks=None,mode='text',style_function_dict=dict(),
                            reg_method='orthog',thresh=None,w_thresh=None,cts_color=True, weights_also=True):
    """
    Here Zref dims (n,K) is set of reference variates
    masks: either None or a tuple (mask_x,mask_y,mask_u,mask_v)
    """
    # load data, and original estimated weights and variates
    d = solp.data
    Ue,Ve,_ = solp.load_est_n_t(pen,nearest=True)

    # restrict to relevant indices and register
    # before registration, inds may be bigger than Zref.shape[1], 
    # transformer map can therefore reduce the dimension
    Ur, Vr = Ue[:,inds], Ve[:,inds]
    Zx, Zy = d.X@Ur, d.Y@Vr
    if Zref is not None:
        transformer_map = utils.register_general(Zx,Zref,reg_method=reg_method,output='transformer_map')
        Ur,Vr,Zx,Zy = map(transformer_map,[Ur,Vr,Zx,Zy])

    # get masks if doing individual masking rather than shared masking
    if masks is None:
        # WARNING! currently arbitrarily take Zx for masking variables; this is bad...
        # think the change here would be to have two different versions, one for Zx, one for Zy TODO
        print(thresh, w_thresh)
        mask_x,mask_y, = get_masks_loading(d,Zx,thresh)    #get_masks(d,Zx,Ur,Vr,thresh,w_thresh)
        if weights_also: mask_u,mask_v = get_masks_weights(Ur,Vr,w_thresh)
    else:
        mask_x,mask_y,mask_u,mask_v = masks

    # define functions to generate the scatter plots, via helpers
    # specify colorscale
    colorscale = 'Bluered_r' if cts_color else 'HSV'
    def _str_corr_scatter(vbls,labs,mask,vrts,style_fns):
        """ Compute structure correlations on masked variables and return scatter plot."""
        masked_labs = labs[mask]
        marker_dict = apply_style_functions(masked_labs,style_fns,colorscale)

        masked_vbls = vbls[:,mask]
        masked_str_corrs = utils.col_corr(masked_vbls,vrts)

        K = vrts.shape[1]
        if K == 2:
            scatter = go.Scatter(
                    name=masked_labs.name,
                    x=masked_str_corrs[:,0],
                    y=masked_str_corrs[:,1],
                    text=masked_labs,
                    mode=mode,
                    textfont=dict(color=color),
                    marker=marker_dict
                )
        elif K == 3:
            scatter = go.Scatter3d(
                    name=masked_labs.name,
                    x=masked_str_corrs[:,0],
                    y=masked_str_corrs[:,1],
                    z=masked_str_corrs[:,2],
                    text=masked_labs,
                    mode=mode,
                    textfont=dict(color=color),
                    marker=marker_dict
                )
        else:
            ValueError('vrts must have 2 or 3 columns')
        return scatter
    
    def structure_correlation_scatter(vbl: str, vrt: str):
        """ process the string inputs and calls _str_corr_scatter
        vbl is either 'X' or 'Y' and vrt is either 'Zx' or 'Zy'
        """
        vrts = Zx if vrt=='Zx' else Zy
        if vbl=='X':
            vbls, labs, mask, style_fns = d.X, d.X_labs, mask_x, style_function_dict['xmarkers']
        elif vbl=='Y':
            vbls, labs, mask, style_fns = d.Y, d.Y_labs, mask_y, style_function_dict['ymarkers']
        else:
            ValueError('vbl must be X or Y')
        return _str_corr_scatter(vbls, labs, mask, vrts, style_fns)
    
    def _weight_scatter(wgts,labs,mask,style_fns):
        """ Return scatter plot for masked weights."""
        K = wgts.shape[1]
        assert K==2, 'only 2D weight plots currently supported'
        # should be straightforward to extend to 3D, but choice of extension depends on downstream needs, 
        # and no downstream need yet, so not yet implemented

        masked_labs = labs[mask]
        marker_dict = apply_style_functions(masked_labs,style_fns,colorscale)

        masked_wgts = wgts[mask,:]

        scatter = go.Scatter(
                name=masked_labs.name,
                x=masked_wgts[:,0],
                y=masked_wgts[:,1],
                text=masked_labs,
                mode=mode,
                textfont=dict(color=color),
                marker=marker_dict
            )
        return scatter

    def weight_scatter(wgt: str):
        """ Processes the string inputs and calls _weight_scatter.
        wgt is either 'U' or 'V'
        """
        # note we are inheriting many python variables from parent environment
        if wgt=='U':
            wgts,labs,mask,style_fns = Ur,d.X_labs,mask_u,style_function_dict['xmarkers']
        elif wgt=='V':
            wgts,labs,mask,style_fns = Vr,d.Y_labs,mask_v,style_function_dict['ymarkers']
        else:
            ValueError('wgt must be U or V')
        
        return _weight_scatter(wgts,labs,mask,style_fns)

    # finally, generate the dictionary of scatter plots
    out_dict=dict()
    for vbl,color in zip(['X','Y'],['red','blue']):
        for vrt in ['Zx','Zy']:
            out_dict[(vrt,vbl)] = structure_correlation_scatter(vbl,vrt)
    if weights_also:
        for wgt,color in zip(['U','V'],['red','blue']):
            out_dict[wgt] = weight_scatter(wgt)

    # also helpful to return the registered variates
    out_dict['Zx'] = Zx
    out_dict['Zy'] = Zy        

    return out_dict


def get_masks_loading(d,Ze,thresh):
    """
    Ue,Ve have only contain indices of interest; same k as for Ze
    d is a data object
    """
    mask_x,_ = utils.threshold_corrs_mat_only(d.X,Ze,thresh=thresh)
    mask_y,_ = utils.threshold_corrs_mat_only(d.Y,Ze,thresh=thresh)
    return mask_x,mask_y

def get_masks_weights(Ue, Ve, w_thresh):
    mask_u = utils.get_l1_mask(Ue,thresh=w_thresh)
    mask_v = utils.get_l1_mask(Ve,thresh=w_thresh)
    return mask_u,mask_v


def get_reference(solp,pen,inds,thresh,w_thresh):
    Zref = solp.load_vrts(pen=pen,inds=inds)[0]
    d = solp.data
    mask_x,_ = utils.threshold_corrs_mat_only(d.X,Zref,thresh=thresh)
    mask_y,_ = utils.threshold_corrs_mat_only(d.Y,Zref,thresh=thresh)
    Ue,Ve,te = solp.load_est_n_t(pen=pen,nearest=True)
    mask_u = utils.get_l1_mask(Ue[:,inds],thresh=w_thresh)
    mask_v = utils.get_l1_mask(Ve[:,inds],thresh=w_thresh)
    return Zref,mask_x,mask_y,mask_u,mask_v


# Applications of this main function in specific cases:
## Makes most sense to consider different penalties for different algorithms, in a static plot
def side_by_side_misc_biplots(dataset,algo_pen_inds_tuples,ref_idx,vrt,                 # positional arguments
                              reg_method='orthog',mask_type='shared',                   # kwargs for registration (sensitive!)
                              thresh=0.5,w_thresh=0.1,vbls=['X','Y'],wgts=['U','V'],    # kwargs to filter what is plotted
                              mode='text',style_function_dict=dict(),cts_color=True):   # kwargs for plot styling
    """ Generate plotly figure of side by side biplots.
    The figure will have two rows: the first row for loadings, the second row for weights
    Different algorithms will be in different columns, with reference in column ref_idx + 1; columns are 1-indexed
    
    Parameters
    ----------
    dataset : str
        name of dataset
    algo_pen_inds_tuples : list of tuples of the form (algo,pen,inds)
        where algo is string, pen is numeric and inds is list of integers
    ref_idx : int
        index of reference algorithm in algo_pen_inds_tuples
    vrt : {'Zx','Zy'}
        string specifying which variate to use to generate biplots
    
    reg_method : str
        method for registration; see utils.register_general for options
    mask_type : {'shared','individual'}
        if shared generate masks for reference algorithm and use that for all other algorithms
        if individual then generate masks for each algorithm separately   
    
    thresh : float
        threshold for masking correlations (using 2-norm of correlation vector so circle on biplot)
    w_thresh: float
        threshold for masking weights (using 1-norm of weight vector so diamond on weight plot)
    vbls : list of {'X','Y'}
        which variables to plot e.g. ['X'] just plots 'X' variables to make plots less cluttered
    wgts : list of {'U','V'}
        which weights to plot

    mode : {'text','markers','markers+text'}
        mode for scatter plot, see plotly documentation for details
    style_function_dict : dict
        dictionary of functions to style the scatter plots
        see get_trs_load_n_weights for details of usage
        examples are given in datasets.py
    cts_color : bool
        determine colour scale used for scatter plots, see get_trs_load_n_weights for details of usage
    """
    # In code below tuple always refers to an (algo,pen,inds) tuple

    # 0. data once to save processing it multiple times
    data = get_dataset(dataset)
    def get_pen_inds_fullp(tuple):
        algo,pen,inds = tuple
        return pen, inds, get_cv_obj_from_data(data,algo).full_path

    # 1. compute reference variates and masks
    pen_ref,inds_ref,solp_ref = get_pen_inds_fullp(algo_pen_inds_tuples[ref_idx])
    Zref,mask_x,mask_y,mask_u,mask_v = get_reference(solp_ref,pen=pen_ref,inds=inds_ref,thresh=thresh,w_thresh=w_thresh)
    masks = (mask_x,mask_y,mask_u,mask_v) if mask_type=='shared' else None
    
    # 2. define utility to get traces from tuple
    def get_trs(tup):
        """Unpack tuple then call get_trs_load_n_weights."""
        pen,inds,fullp = get_pen_inds_fullp(tup)
        trs = get_trs_load_n_weights(fullp,pen=pen,inds=inds,Zref=Zref,
                                    masks=masks,thresh=thresh,w_thresh=w_thresh,
                                    mode=mode,style_function_dict=style_function_dict,reg_method=reg_method,
                                    cts_color=cts_color)
        return trs

    # 3. define utility to generate an initial labelled figure from list of tuples
    def setup_figure(tuples):
        num_cols = len(tuples)
        def get_str(tup): 
            algo, pen, inds = tup
            return f"{algo_labels[algo]}, {'{:.1e}'.format(pen)}, {list(inds)}"
        subplot_titles = tuple([get_str(tup) + ", s.c.s" for tup in tuples]
                            + [get_str(tup) + ", wt.s" for tup in tuples])
        fig = make_subplots(rows=2, cols=num_cols,shared_yaxes=True,
                            subplot_titles=subplot_titles
                            )
        # Add in circles and diamonds at relevant thresholds
        circle = utils.gen_circle()
        diamond = utils.gen_diamond()
        for col in range(1,num_cols+1):
            fig.update_xaxes(scaleanchor = "y",scaleratio = 1,row=1,col=col)
            fig.add_trace(go.Scatter(x=circle['x'],y=circle['y'],name='|x|=1',line={'color':'blue'})
                        ,row=1,col=col)
            fig.add_trace(go.Scatter(x=thresh*circle['x'],y=thresh*circle['y'],
                                    name=f'|x|={thresh}',line={'color':'red'})
                        ,row=1,col=col)

            fig.add_trace(go.Scatter(x=w_thresh*diamond['x'],y=w_thresh*diamond['y'],
                                    name=f'|x|_1={w_thresh}',line={'color':'gray'})
                        ,row=2,col=col)
        return fig

    # 4. generate figure then add the interesting traces
    fig = setup_figure(algo_pen_inds_tuples)
    for idx,tup in enumerate(algo_pen_inds_tuples):
        col = idx + 1 # column indexing starts from 1
        trs = get_trs(tup)
        for vbl in vbls:
            fig.add_trace(trs[(vrt,vbl)],row=1,col=col)
        for wgt in wgts:
            fig.add_trace(trs[wgt],row=2,col=col)

    return fig

def rescale_slider_comparison(fig: go.Figure,weight_x_max=1,weight_y_max=1,width=1000,height=800):
    fig.update_layout(
        autosize=True,
        width=width,
        height=height)
    for col in [1,2,3]:
        fig.update_xaxes(range=[-weight_x_max,weight_x_max],row=2,col=col)
        fig.update_yaxes(range=[-weight_y_max,weight_y_max],row=2,col=col)
    return fig


# Created this function to mirror the call from utils.biplot_w_ref but hopefully be much clearer
# Note that the some of the key word arguments have slightly different names to that older version
def biplot_3D(solp,pen,inds,
                 ref_vrts=None,vrt='Zx',reg_method='perms_n_signs',
                 vbls=['X','Y'],thresh=0.6,
                 in_sphr=False, mode='markers',cts_color=True,style_function_dict=dict()):
    """
    Plot a biplot, from given reference variates
    
    Parameters
    ----------
    solp : SolutionPath
        Solution path object of interest
    pen : float
        Penalty of interest
    inds : list of integers
        Indices of variates to consider
    ref_vrts : np.array
        Reference variates, if None then use original inds without registration
    vrt : {'Zx','Zy'}
        Which variates to use
    reg_method : str
        method for registration; see utils.register_general for options
    
    vbls : list of {'X','Y'}
        which variables to plot e.g. ['X'] just plots 'X' variables to make plots less cluttered
    thresh : float
        threshold for masking correlations (using 2-norm of correlation vector so circle on biplot)
    in_sphr : bool
        whether to plot an inner thresholding sphere (with radius thresh)
    
    mode : {'text','markers','markers+text'}
        mode for scatter plot, see plotly documentation for details
    style_function_dict : dict
        dictionary of functions to style the scatter plots
        see get_trs_load_n_weights for details of usage
        examples are given in datasets.py
    cts_color : bool
        determine colour scale used for scatter plots, see get_trs_load_n_weights for details of usage
    """
    # 1. create traces for the plot
    trs = get_trs_load_n_weights(solp,pen,inds,Zref=ref_vrts,
                                 masks=None,mode=mode, style_function_dict=style_function_dict,
                                 reg_method=reg_method, thresh=thresh, cts_color=cts_color,
                                 weights_also=False)
    
    # 2. create the figure and add the background traces 
    fig = go.Figure()
    if in_sphr:
            sphere = utils.my_sphere(0,0,0,radius=thresh)
            fig.add_trace(sphere)

    # 3. add the traces of interest
    for vbl in vbls:
        fig.add_trace(trs[(vrt,vbl)])
    
    # to facilitate registration between multiple biplots, also return the variates
    # (I adopted this in some earlier notebooks, but not elegant, and a good target for refactoring)
    return trs[vrt], fig



