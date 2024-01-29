import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils

def plot_histogram(X,Y):
    fig,axs = plt.subplots(ncols=2,figsize=(9,4))
    plt.subplots_adjust(wspace=.3)
    plots = [sns.kdeplot(X[:,j],ax=axs[0],label=f'j={j}') for j in range(5)]
    #axs[0].set_xlim([-3,3])
    axs[0].set_title('Histograms of X_j')
    axs[0].legend()
    plots = [sns.kdeplot(Y[:,j],ax=axs[1],label=f'j={j}') for j in range(5)]
    #axs[1].set_xlim([-5,5])
    axs[1].set_title('Histograms of Y_j')
    axs[1].legend()
    return axs

def bubble_plot(u,ax=None,color='green'):
    if ax is None: fig,ax=plt.subplots()
    x = np.arange(len(u))
    markerline, stemlines, baseline = ax.stem(x,u, linefmt ='grey', use_line_collection = True)
    markerline.set_markerfacecolor('none')
    markerline.set_markeredgecolor(color)
    return markerline
    #ax.set_ylim([-1,1])

# plot X-view and Y-view canonical directions side by side
def side_by_side(axrow,Uh,Vh,K=None):
    if K is None: K = Uh.shape[1]
    cmap = plt.get_cmap('viridis')
    U_lines = [bubble_plot(Uh[:,k],axrow[0],color=cmap.colors[k*255//K]) for k in range(K)]
    V_lines = [bubble_plot(Vh[:,k],axrow[1],color=cmap.colors[k*255//K]) for k in range(K)]
    axrow[0].legend(U_lines,list(range(1,K+1)),ncol=K)
    axrow[1].legend(V_lines,list(range(1,K+1)),ncol=K)
    return axrow

# additionally plot a bubble plot showing how the canonical correlations decay
def tri_plot(Sig,m):
    """Bubble-plots first 40 can. corrs and first 4 can. directions"""
    fig,axs = plt.subplots(ncols=3,figsize=(10,4))
    U,V,D = utils.cca_from_cov_mat(Sig,m)
    side_by_side(axs[1:],U[:,:4],V[:,:4])
    bubble_plot(D[:40],ax=axs[0])
    axs[0].set_title('Canonical correlations')
    axs[1].set_title('View1 directions')
    axs[2].set_title('View2 directions')
    return axs