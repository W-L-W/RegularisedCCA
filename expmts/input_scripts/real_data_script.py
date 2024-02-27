import numpy as np
import os

from src.algos import get_pens
from src.scaffold.wrappers import get_cv_obj_from_data, compute_everything, get_dataset, get_cv_object
from src.scaffold.io_preferences import save_mplib, save_plotly, output_folder_real_data
from src.plotting.comparison import (
    row_plot3, 
    stab_row_plot, 
    corr_decay_misc, 
    traj_stab_array, 
    pretty_arr_plot,
    sq_overlap_misc,
    sq_overlap_folds,
    sq_overlap_path,
)
from src.plotting.biplots import biplot_3D, side_by_side_misc_biplots, rescale_slider_comparison
from real_data.styling import dset_abbrev, mb_2d_style_fns, mb_3d_style_fns, bd_3d_style_fns, kumar_style_fns
# parameters for this script
# to move to kwargs later so that can be run from command line

dataset = 'breastdata'


# wit, i.e. sPLS, is not a genuine CCA algorithm, and gglasso scales too poorly to fit on the full breastdata set
algos_all = ['wit','suo','gglasso','ridge']
algos_CCA = ['suo','gglasso','ridge']
algos_bd = ['wit','suo','ridge']
algos_CCA_bd = ['suo','ridge']

algos_to_fit = {'nutrimouse': algos_all, 'microbiome': algos_all, 'breastdata': algos_bd}
algos = algos_to_fit[dataset]
algos_to_traj_comp = {'nutrimouse': algos_CCA, 'microbiome': algos_CCA, 'breastdata': algos_bd}
algos_tc = algos_to_traj_comp[dataset]
data = get_dataset(dataset)


# then process the output and make the plots
def get_pen_trios(algos, data, recompute=False):
    file_name = output_folder_real_data(dataset, mode='processed') + 'pen_trios.npz'
    print(file_name)
    # if file doesn't exist compute pen trio
    if not os.path.exists(file_name) or recompute:
        def get_trio(scale): return float(scale)**np.array([-1,0,1])
        def get_best_pen(algo): return get_cv_obj_from_data(data,algo).get_best_pen('r2s5')
        pen_trios = {}
        for algo in algos:
            pen_trios[algo] = get_best_pen(algo) * get_trio(3)
        # save this dictionary to file
        # create the parent directory if required
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.savez(file_name, **pen_trios)
    else:
        pen_trios = np.load(file_name)
        return pen_trios

def biplot2D_microbiome_bonus():
    algos = ['suo', 'gglasso', 'ridge'] # good to have gglasso in middle
    best_pairs = [(algo, pen_trios[algo][1]) for algo in algos]

    def get_inds(algo): return [0,1] if algo=='gglasso' else range(3)
    tuples = [(*p,get_inds(p[0])) for p in best_pairs]

    fig = side_by_side_misc_biplots('microbiome', tuples, ref_idx=1,vrt='Zx',
                                    reg_method='orthog', mask_type='shared',
                                    vbls=['X','Y'],wgts=['U','V'],thresh=0.4,w_thresh=0.07,
                                    mode='markers',style_function_dict=mb_2d_style_fns,
                                    )
    rescale_slider_comparison(fig.update_traces(),0.4,0.6)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=50, b=50, t=50,),
    )
    save_plotly(fig, 'mb_biplot_2D_grid')

def biplot3D_microbiome_kumar():
    fullp = get_cv_object('microbiome','gglasso').full_path
    pen = pen_trios['gglasso'][1]
    _, fig = biplot_3D(fullp, pen, inds=[0,1,2], 
                         ref_vrts=None, vrt='Zx', reg_method='orthog',
                         vbls=['X','Y'], thresh=0.4,
                         in_sphr=True, mode='markers', cts_color=True, style_function_dict=kumar_style_fns,
                         )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        title=False,
        height=600,
        scene_camera=dict(up=dict(x=0, y=0, z=1), 
                          eye=dict(x=1.3, y=1.3, z=1), 
                          center=dict(x=0, y=0, z=-0.2))
    )
    save_plotly(fig, 'mb_biplot_3D_kumar_annotations')

def biplot3D_microbiome_bonus():
    algos_ggl_first = ['gglasso', 'ridge', 'suo']
    best_pairs = [(algo, pen_trios[algo][1]) for algo in algos_ggl_first]

    vrts = dict()
    figs = dict()

    for algo,pen in best_pairs:
        fullp = get_cv_object('microbiome',algo).full_path
        if algo == 'gglasso': ref_vrts = None
        else: ref_vrts = vrts['gglasso']

        vrts[algo],figs[algo] = biplot_3D(fullp,pen,inds=[0,1,2],
                                          ref_vrts=ref_vrts, vrt='Zx', reg_method='orthog',
                                          vbls=['Y'], thresh=0.4,
                                          in_sphr=True, mode='markers', cts_color=True, style_function_dict=mb_3d_style_fns,
                                          )
    def custom_layout_updater(fig):
        x_lim = 0.8; y_lim = 0.7; z_lim = 0.6
        return fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=0), # remove all margins
            title=False,
            height=600,
            scene_camera = dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=1, y=1, z=0.7),
                center=dict(x=0, y=0, z=-0.12),
            ),
            scene = dict(
                aspectmode='manual',
                aspectratio=dict(x=x_lim, y=y_lim, z=z_lim),
                xaxis=dict(range=[-x_lim, x_lim]),
                yaxis = dict(range=[-y_lim,y_lim]),
                zaxis = dict(range=[-z_lim,z_lim]),
            ),
        )
        
    for algo,_ in best_pairs:
        custom_layout_updater(figs[algo])
        save_plotly(figs[algo], f'mb_biplot_3D_{algo}')

def biplot3D_breastdata_bonus():
    # to fill in; coppy over from essay.ipynb TODO next
    algos = ['suo', 'ridge']
    best_pairs = [(algo, pen_trios[algo][1]) for algo in algos]

    vrts = dict()
    figs = dict()

    for algo,pen in best_pairs:
        fullp = get_cv_object('breastdata',algo).full_path
        if algo == 'suo': ref_vrts = None
        else: ref_vrts = vrts['suo']

        vrts[algo],figs[algo] = biplot_3D(fullp,pen,inds=[0,1,2],
                                          ref_vrts=ref_vrts, vrt='Zx', reg_method='orthog',
                                          vbls=['X'], thresh=0.4,
                                          in_sphr=True, mode='markers', cts_color=False, style_function_dict=bd_3d_style_fns,
                                          )
        
    def custom_layout_updater(fig):
        return fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=0), # remove all margins
            title=False,
            height=600,
            scene_camera = dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=-1.5, y=-1.5, z=1),
                center=dict(x=0, y=0, z=-0.2),
            ),
            scene = dict(
                xaxis=dict(range=[-1,1]),
                yaxis = dict(range=[-1,1]),
                zaxis = dict(range=[-1,1]),
            )
        )
    
    for algo,_ in best_pairs:
        custom_layout_updater(figs[algo])
        save_plotly(figs[algo], f'bd_biplot_3D_{algo}_coloured')





if __name__ == '__main__':
    recompute = False
    if recompute:
        #first fit all the algorithms
        for algo in algos:
            print(f'Fitting {algo}')
            cv_obj = get_cv_obj_from_data(data,algo)
            pens = get_pens(algo, data.n, mode='run')
            print(pens)
            compute_everything(cv_obj,pens)

    pen_trios = get_pen_trios(algos, data)
    print('loaded pen trios', pen_trios)

    # # initial correlation and stability along trajectories
    # fig, _, _ = row_plot3(data, algos, lambda x: x**2,[0,2,4],y_label='r2sk-cv')
    # save_mplib(fig, f'{dset_abbrev[dataset]}_traj_corr')
    # criteria = ['wt_U', 'vt_U']
    # fig, _ = stab_row_plot(data, algos, criteria, [1,3,5])
    # save_mplib(fig, f'{dset_abbrev[dataset]}_traj_stab')

    # correlation decay
    best_pairs = [(algo, pen_trios[algo][1]) for algo in algos]
    fig = corr_decay_misc(dataset, best_pairs)
    save_mplib(fig, f'{dset_abbrev[dataset]}_corr_decay_best_pens')

    # trajectory comparison matrices
    algo_penlist_pairs = [(algo,pen_trios[algo]) for algo in algos_tc]
    arr_wt = traj_stab_array(dataset,algo_penlist_pairs,inds=[0,1,2],space='weight')
    arr_vt = traj_stab_array(dataset,algo_penlist_pairs,inds=[0,1,2],space='variate')
    fig_wt, _ = pretty_arr_plot(arr_wt[:,:,2], algo_penlist_pairs)
    fig_vt, _ = pretty_arr_plot(arr_vt[:,:,2], algo_penlist_pairs)
    save_mplib(fig_wt, f'{dset_abbrev[dataset]}_traj_comp_wt')
    save_mplib(fig_vt, f'{dset_abbrev[dataset]}_traj_comp_vt')

    # overlap matrices
    fig, _ = sq_overlap_misc(dataset, best_pairs, ref_pair_idx=1)
    save_mplib(fig, f'{dset_abbrev[dataset]}_sq_overlap_best_pens')

    if dataset == 'microbiome':
        # # bonus plots for microbiome appendix
        # algo = 'gglasso'; folds=[0,1,2]; pens = pen_trios[algo]
        # fig, _ = sq_overlap_folds(dataset,algo,folds,ref_pen=pens[1],inds=range(5))
        # save_mplib(fig, f'{dset_abbrev[dataset]}_sq_overlap_gglasso_folds')

        # best_pairs = [(algo, pen_trios[algo][1]) for algo in algos]
        # fig, _ = sq_overlap_misc(dataset, best_pairs, ref_pair_idx=1, reg='orthog')
        # save_mplib(fig, f'{dset_abbrev[dataset]}_sq_overlap_best_pens_orthog')

        # suo_pens = pen_trios['suo']
        # fig, _ = sq_overlap_path(dataset, 'suo',suo_pens,ref_pen=suo_pens[1],reg='signs')
        # save_mplib(fig, f'{dset_abbrev[dataset]}_sq_overlap_suo_path')

        # biplot3D_microbiome_bonus()

        # biplot3D_microbiome_kumar()
        biplot2D_microbiome_bonus()

    elif dataset == 'breastdata':
        biplot3D_breastdata_bonus()



# progress at lunch break: need to fix error on console about get_best_pen
        
        
