import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
from gglasso.helper.basic_linalg import adjacency_matrix

def scale(mat,power):
    return np.sign(mat) * np.abs(mat)**power

def plot_graph_from_precision(Theta,ax=None,simple=True):
    A = adjacency_matrix(Theta)

    G = nx.from_numpy_array(A)
    if simple: pos = nx.drawing.layout.spring_layout(G, seed = 1234)
    else: pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    #pos = nx.drawing.layout.spring_layout(G, seed = 42)

    if ax is None: fig,ax = plt.subplots(figsize=(8,8))
    nx.draw_networkx(G, ax=ax, pos = pos, node_color = "darkblue", node_size=100,
    edge_color = "darkblue", font_color = 'white', font_size=8,with_labels = True)

def plot_graph_and_heatmap(Thetas,labels=None,scale=8,simple=True):
    num_thetas = len(Thetas)
    fig, axs = plt.subplots(num_thetas,2, figsize=(2*scale,num_thetas*scale))

    for r in range(num_thetas):
        Theta = Thetas[r]
        plot_graph_from_precision(Theta,axs[r,0],simple=simple)
        sns.heatmap(Theta, cmap = "coolwarm", linewidth = .5, square = True, vmin = -1,vmax=1, center = 0, cbar = True, \
                xticklabels = [], yticklabels = [], ax = axs[r,1])
        if labels is not None:
            axs[r,0].set_title(labels[r])
            axs[r,1].set_title(labels[r])

def plot_graph_and_heatmap_T(Thetas,labels=None,scale=8,simple=True):
    num_thetas = len(Thetas)
    fig, axs = plt.subplots(2,num_thetas, figsize=(num_thetas*scale,2*scale))

    for r in range(num_thetas):
        Theta = Thetas[r]
        plot_graph_from_precision(Theta,axs[0,r],simple=simple)
        sns.heatmap(Theta, cmap = "coolwarm", linewidth = .5, square = True, cbar = True,center=0, \
                xticklabels = [], yticklabels = [], ax = axs[1,r]) #vmin = -0.5, vmax = 0.5
        if labels is not None:
            axs[0,r].set_title(labels[r])
            axs[1,r].set_title(labels[r])

# without graph
def plot_heatmap_pairs(Thetas,notherdim=2,labels=None,scale=8,transpose=False):
    """ Thetas a list of covariance matrices; labels a list of strings of the same length"""
    num_thetas = len(Thetas)
    if transpose:
        fig, axs = plt.subplots(ncols=num_thetas,nros=notherdim, figsize=(num_thetas*scale,notherdim*scale))
    else:
        fig, axs = plt.subplots(nrows=num_thetas, ncols=notherdim,figsize=(scale*notherdim,num_thetas*scale))


    if num_thetas == 1: axs=[axs] #syntax fix

    for r in range(num_thetas):
        Th1,Th2 = Thetas[r]
        ax1,ax2 = axs[r,:] if transpose==False else axs[:,r]
        sns.heatmap(Th1, cmap = "coolwarm", linewidth = .5, square = True, center = 0, cbar = True, \
                xticklabels = [], yticklabels = [], ax = ax1)
        sns.heatmap(Th2, cmap = "coolwarm", linewidth = .5, square = True, center = 0, cbar = True, \
                xticklabels = [], yticklabels = [], ax = ax2)
        if labels is not None:
            [ax.set_title(labels[r]) for ax in (ax1,ax2)]