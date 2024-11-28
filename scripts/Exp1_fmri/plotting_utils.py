import itertools
import matlab.engine
import plotly.express as px
import plotly.graph_objs as go
import plotly

import drs
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time
import pandas as pd
import glob
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump,load
import plotly.express as px

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'src'))
from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import compute_rdm,lower_tri,upper_tri, compute_rdm_nomial, compute_rdm_identity
from zpyhelper.MVPA.preprocessors import scale_feature, average_odd_even_session,normalise_multivariate_noise, split_data, concat_data
from zpyhelper.MVPA.estimators import PatternCorrelation, MultipleRDMRegression, NeuralRDMStability

#from multivariate.modelrdms import ModelRDM
from multivariate.modelrdms import ModelRDM
from multivariate.mvpa_runner import RSARunner
from utils.composition_modelfit import multi_start_optimisation

import sklearn
from sklearn.manifold import MDS,TSNE 
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score,r2_score

import scipy
eng = matlab.engine.start_matlab()

def plot_rdm_withlabel(rdm_mat:np.ndarray,
                        lower_tri_only=True,
                        reorder_x=None,reorder_y=None,
                        xticks = None,yticks=None,
                        stimlabels_x=None,stimlabels_y=None,
                        stimgroups=None,colors:dict=None,
                        highlightzone=None,
                        annot=False,
                        ax:matplotlib.axes.Axes=None):
    """plot rdm using heatmap with option to do labeling, highlighting, and reordering etc 

    Parameters
    ----------
    rdm_mat : np.ndarray
        matrix to plot with heatmap
    lower_tri_only : bool, optional
        plot the whole matrix or only the lower triagular part, by default True
    reorder_(x or y): array_like,optional
        reorder rows and columns of the matrix for ploting
    xticks : arry_like, optional
        xticks lockation, by default None
    yticks : arry_like, optional
        yticks lockation, by default None
    stimlabels_x : arry_like, optional
        xticklabels (already reordered), by default None
    stimlabels_y : arry_like, optional
        yticklabels (already reordered), by default None
    stimgroups : arry_like, optional
        groups of stimuli, by default None
    colors : dict, optional
        colors used for different groups of stimuli, by default None
    highlightzone : list, optional
        a list of the form [tuple,int,int], by default None
                +------------------+
                |                  |
              height               |
                |                  |
               (xy)---- width -----+
    ax : matplotlib.axes.Axes, optional
        the axis to plot on, if None will create a new axis, by default None

    Returns
    -------
    matplotlib.pyplot axs
        axis handle for the plot
    """
    rorder_mat = deepcopy(rdm_mat)
    reorder_x = np.arange(rdm_mat.shape[1]) if reorder_x is None else reorder_x
    reorder_y = np.arange(rdm_mat.shape[0]) if reorder_y is None else reorder_y
    assert np.size(reorder_x) == rdm_mat.shape[1]
    assert np.size(reorder_y) == rdm_mat.shape[0]
    rorder_mat = np.array([col[reorder_x] for col in rorder_mat.T])
    rorder_mat = np.array([row[reorder_y] for row in rorder_mat])
    
    if stimgroups is not None:
        sg_x = np.array(stimgroups)
        sg_y = np.array(stimgroups)
    # if stimlabels_x is not None:
    #     stimlabels_x = np.array(stimlabels_x)[reorder_x]
    # if stimlabels_y is not None:
    #     stimlabels_y = np.array(stimlabels_y)[reorder_y]
    # if stimgroups is not None:
    #     sg_x = np.array(stimgroups)[reorder_x]
    #     sg_y = np.array(stimgroups)[reorder_y]
    
    if lower_tri_only:
        rdm_mat = np.full_like(rorder_mat,fill_value=np.nan)
        rdm_mat[lower_tri(rorder_mat)[1]] = lower_tri(rorder_mat)[0]
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    sns.heatmap(rdm_mat,ax=ax,annot=annot)
    
    if xticks is not None:
        ax.set_xticks(ticks=xticks, labels=stimlabels_x,rotation=90)
    
    if yticks is not None:
        ax.set_yticks(ticks=yticks, labels=stimlabels_y,rotation=0)
    
    if colors is not None:
        for xtick, g in zip(ax.get_xticklabels(), sg_x):
            xtick.set_color(colors[g])
        for ytick, g in zip(ax.get_yticklabels(), sg_y):
            ytick.set_color(colors[g])
    ##highlight the odd-even run correlation bits in the rdm
    if highlightzone is not None:
        xy,width,height = highlightzone
        ax.add_patch(
        patches.Rectangle(
            xy,
            width,
            height,
            edgecolor='red',
            fill=False,
            lw=4
        ) )
    return ax   


def plot_rdm_mds(rdm:np.ndarray,ncompo:int,
                 stimdf:pd.DataFrame,gcol:str,huecol:str,
                 stylecol:str,stim_namer:callable=None,plot = True,plot_3d=True,mds_seed=None):
    """plot rdm and mds based on rdm

    Parameters
    ----------
    rdm : numpy.ndarray
        rdm matrix
    ncompo : int
        number of component for mds
    stimdf : pd.DataFrame
        dataframe containing stimuli information
    gcol : str
        column in stimdf used to group stimuli, different groups will be assigned different colors in x tick label in heatmap of rdm
    huecol : str
        column in stimdf used for labeling in mds plot
    stylecol : str
        column in stimdf used for labeling in mds plot
    plot_3d : bool, optional
        plot 3d mds plot, by default True

    Returns
    -------
    tuple
        fig, np.array([ax1,ax2]), X_df
    """
    if np.logical_and(plot_3d,ncompo<3):
        plot_3d = False

    matlabeng = matlab.engine.start_matlab()
    X_transformed = np.array(matlabeng.cmdscale(rdm,ncompo)) 
    matlabeng.quit()
    #X_transformed =  MDS(n_components=ncompo,metric=False,dissimilarity='precomputed',normalized_stress='auto', n_init=500, random_state=mds_seed).fit_transform(rdm)
    if ncompo>X_transformed.shape[1]:
        print("MDS returning fewer components than required, remaining dimensions are filled with zero")
        fill_zeros = np.zeros((X_transformed.shape[0],ncompo-X_transformed.shape[1]))
        X_transformed = np.concatenate([X_transformed,fill_zeros],axis=1)
    X_df = pd.concat([
        pd.DataFrame(X_transformed,
                columns=[f"MDS ax{j+1}" for j in range(ncompo)]),
        stimdf],axis=1)

    lg = np.array(stimdf[gcol])
    lg_palette = dict(zip(np.unique(lg),
                          sns.color_palette("colorblind",np.unique(stimdf[gcol]).size)))
    if stim_namer is None:
        X_df["stim_name"] = [f"x{'%d' % x}y{'%d' % y}" for x,y in zip(np.array(X_df[huecol]),np.array(X_df[stylecol]))]
    else:
        X_df["stim_name"] = [stim_namer(x,y) for x,y in zip(np.array(X_df[huecol]),np.array(X_df[stylecol]))]

    if plot:
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(121)

        plot_rdm_withlabel(rdm,lower_tri_only=True,stimgroups=lg,
                                xticks=np.arange(rdm.shape[0])+.5,yticks=np.arange(rdm.shape[0])+.5,
                                stimlabels_x=np.array(X_df["stim_name"]),
                                stimlabels_y=np.array(X_df["stim_name"]),
                                colors=lg_palette,ax=ax1)
        if plot_3d:
            ax2 = fig.add_subplot(122,projection='3d')
            plt_arr = np.array(X_df[["MDS ax1","MDS ax2","MDS ax3",gcol,"stim_name"]])
            for j,arr in enumerate(plt_arr):
                m1,m2,m3,sg,sname = arr
                ax2.text(m1,m2,m3,sname, color=lg_palette[sg])
            ax2.set_xlabel("MDS ax1")
            ax2.set_ylabel("MDS ax2")
            ax2.set_zlabel("MDS ax3")
            axlim_x = np.min(plt_arr[:,:2]), np.max(plt_arr[:,:2])
            axlim_y = np.min(plt_arr[:,:2]), np.max(plt_arr[:,:2])
            axlim_z = np.min(plt_arr[:,:2]), np.max(plt_arr[:,:2])
            ax2.set_xlim(*axlim_x)
            ax2.set_ylim(*axlim_y)
            ax2.set_zlim(*axlim_z)
            #plt.legend(*sc.legend_elements(),loc="center",bbox_to_anchor=(1.1,0.5))
        else:
            ax2 =fig.add_subplot(122)
            sns.scatterplot(X_df,x="MDS ax1",y="MDS ax2", hue=huecol,style=stylecol,ax=ax2)
            sns.move_legend(ax2,loc="center",bbox_to_anchor=(1.4,0.5))
        fig.tight_layout()
        return fig, np.array([ax1,ax2]), X_df
    else:
        return X_df

def gen_pval_annot(p,show_pval=0.1):
    asterisks = np.array(["","*","**","***"])
    plevel    = [p>=0.05,all([p<0.05,p>=0.01]),all([p<0.01,p>=0.001]),p<0.001]
    if type(show_pval) == bool:
        if show_pval:
            showpths = 0.1 if show_pval else 0
    elif isinstance(show_pval,float):
        showpths = show_pval           
    else:
        raise Exception("`show_pval` must be float or boolean")
    annot = 'p=%.3f%s' % (p,asterisks[plevel][0]) if p<showpths else asterisks[plevel][0]  
    return annot

def grouped_barscatter_withstats(datadf,
                                 facet_vars,xvar, gvar, yvar, 
                                 prows,pcols,hrow=6,wcol=7,hues=None,
                                 yvartest=None, statfunc=None,
                                 ):
    assert prows*pcols>=len(list(enumerate(datadf.groupby(facet_vars))))
    yvartest = yvar if yvartest is None else yvartest
    assert all([x in datadf.columns for x in [xvar, gvar, yvar, yvartest]])

    if not isinstance(datadf[gvar].dtype, pd.api.types.CategoricalDtype):
        datadf[gvar] = pd.Categorical(datadf[gvar],categories=np.unique(datadf[gvar]))
    group_names = datadf[gvar].cat.categories
    if isinstance(hues, list):
        hue_dict = dict(zip(group_names,hues))
    elif isinstance(hues,dict):
        hue_dict = hues
    else:
        hue_dict = dict(zip(group_names,sns.color_palette(None,len(group_names))))

    if statfunc is None:
        statfunc = lambda yt: scipy.stats.wilcoxon(yt,alternative="greater").pvalue
    
    fig,axes = plt.subplots(prows,pcols,figsize=(wcol*pcols,hrow*prows))
    y_lim = np.array([np.min(datadf[yvar]),np.max(datadf[yvar])])*1.1
        
    for ifacet,(facet_val,facet_df) in enumerate(datadf.groupby(facet_vars)):
        xr = 0.3
        n_group = len(group_names)
        bw = xr*2/n_group
        dodge_r = xr-bw/2
        xdev = list(np.linspace(-dodge_r,dodge_r,n_group,endpoint=True))
        x_centers=np.arange(np.unique(facet_df[xvar]).size)
        x_names = facet_df[xvar].cat.categories
        for kg,(gname,g_df) in enumerate(facet_df.groupby(gvar)):
            x_locs = dict(zip(x_names ,x_centers+xdev[kg]))
            for xname,xloc in x_locs.items():   
                y = g_df[g_df[xvar]==xname][yvar].to_numpy()
                yt = g_df[g_df[xvar]==xname][yvartest].to_numpy()
                jitt_x = np.random.random(y.shape)*.5*dodge_r - .5*dodge_r/2
                axes.flatten()[ifacet].text(x=xloc-dodge_r, y=y_lim[1]/(1.1+0.05*kg),s=gen_pval_annot(statfunc(yt)),color=hue_dict[gname])
                axes.flatten()[ifacet].scatter(x=xloc*np.ones_like(y)+jitt_x,y=y,color=hue_dict[gname],s=5,alpha=0.6)
                axes.flatten()[ifacet].bar(xloc,y.mean(),width=(bw)*0.95,label=gname,
                                    color='None',edgecolor=hue_dict[gname])
                axes.flatten()[ifacet].errorbar(xloc,y.mean(),yerr=y.std()/np.sqrt(y.size),capsize=5,
                                    barsabove=True,
                                    ecolor=hue_dict[gname],alpha=1,linewidth=2)
        axes.flatten()[ifacet].set_xticks(ticks=np.arange(x_names.size),
                                        labels=x_names,
                                        fontweight="bold")
        axes.flatten()[ifacet].set_ylim(y_lim)

        axes.flatten()[ifacet].set_ylabel(yvar,fontweight="bold",fontsize="large")
        axes.flatten()[ifacet].set_title(f"{facet_val}",fontweight="bold",fontsize="x-large")
        axes.flatten()[ifacet].set_xlabel(xvar,fontweight="bold",fontsize="large")
    fig.tight_layout()
    return fig,axes