"""
This file computes the PS and null distribution of PS for the training stimuli that are not the center stimuli.
"""
import numpy as np
import scipy
import pandas as pd


import json
from copy import deepcopy
import os
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri
from zpyhelper.MVPA.preprocessors import split_data

import sys
project_path = r"D:\OneDrive - Nexus365\pirate_ongoing"
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    #fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    #fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

ROIRSAdir = os.path.join(project_path,'AALandHCPMMP1andFUNCcluster')

def cal_PS(X:np.ndarray,stim_locs:np.ndarray, uniqueaxlocs:list)->float:
    """calculate PS for a single run

    Parameters
    ----------
    X : np.ndarray
        activity pattern matrix of the run, shape = (nstim, nvoxel)
    stim_locs : np.ndarray
        locations corresponding to the stimuli in the activity pattern matrix
    uniqueaxlocs : list
        unique axis locations    

    Returns
    -------
    float
        calculated parallelism score (average cosine similarity between axes)
    """
    xlocs, ylocs = stim_locs[:,0], stim_locs[:,1]
    assert all([np.sum(xlocs == j)==1 for j in uniqueaxlocs])
    assert all([np.sum(ylocs == j)==1 for j in uniqueaxlocs])

    # pick x/y stims and put them into same order
    xstims = np.array([X[xlocs== j,:][0] for j in uniqueaxlocs])
    ystims = np.array([X[ylocs == j,:][0] for j in uniqueaxlocs])
    return upper_tri(parallel_axes_cosine_sim(xstims,ystims))[0].mean()

def compute_shuffle(X:np.ndarray,seed=1)->np.ndarray:
    """get row-permuted activity pattern matrix. \n
    During permutation, the correspondence between stimulus and activity pattern is shuffled.

    Parameters
    ----------
    sX : np.ndarray
        the original activity pattern matrix
    
    seed : int, optional
        seed for random generator, by default 1

    Returns
    -------
    float
        parallelism score for permuted activity pattern matrix
    """
    # initialse random generator for reproducibility
    randomgen = np.random.default_rng(seed)
    pX = randomgen.permutation(X)
    return pX

def compute_PSshuffled_for_allruns(Xs:list,stim_locs:list,uniqueaxlocs:list,seed=1)->list:
    """compute PS for permuted activity pattern matrix for all runs

    Parameters
    ----------
    Xs : list
        a list of activity patterns for all runs
    stim_locs : list
        a list of stimuli locations for all runs
    uniqueaxlocs : list
        unique axis locations    
    seed : int, optional
        seed for random generator, by default 1

    Returns
    -------
    list
        a list of computed PS for each run
    """

    n_runs = len(Xs)
    assert len(stim_locs) == n_runs
    PSshuffled_runs = [cal_PS(compute_shuffle(sX,seed=seed+irun*100000),
                                              slocs,
                                              uniqueaxlocs) for irun, (sX,slocs) in enumerate(zip(Xs,stim_locs))]
    return PSshuffled_runs


# Saving Configuration
resoutput_dir = os.path.join(ROIRSAdir,"axisPSsplitrun")
checkdir(resoutput_dir)


### we save: the estimate of PS per run
### and the null distribution of PS per run for each participants
averagePS_savefn = "PS_NCtrainingstim"
runwisePS_savefn = f"{averagePS_savefn}_perrun"


## calculate PS and null distribution of PS from permutation 10000
# define number of permutation
n_perm  = 10000
run_PS_permutation = False
run_classification = True
if run_PS_permutation:
    # Load data
    roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
    rois = [x for x in list(roi_data.keys()) if "bilateral" in x] + ["allgtlocPrecentral_left"]
    ## select activity pattern of non-center training stimuli
    print("Selecting activity pattern of non-center training stimuli")
    sub_patterns = {}
    sub_stimdfs = {}
    for ir, roi in enumerate(rois):
        print(f"{ir+1}/{len(rois)}",end="\n")
        sub_patterns[roi], sub_stimdfs[roi] = [],[]
        for subid,subdata in zip(subid_list,roi_data[roi]):
            
            preprocedX = deepcopy(subdata["preprocX"])
            stimdf = subdata["stimdf"].copy()
            
            navi_filter = stimdf.taskname.to_numpy() == "navigation"

            #average across sessions, so we get a 25*nvoxel matrix
            #navi_X = np.mean(split_data(X      = preprocedX[navi_filter,:],
            #                            groups = stimdf[navi_filter].copy().stim_session.to_numpy()),
            #                axis=0)
            #sub_dfall = stimdf[navi_filter&(stimdf.stim_session==0)].copy().reset_index(drop=True)
            
            navi_X = preprocedX[navi_filter,:]
            sub_dfall = stimdf[navi_filter].copy().reset_index(drop=True)
            
            sub_dfall['training_axsetstr'] = ["x" if axs==0 else "y" for axs in sub_dfall['training_axset']]
            # make loc into integer
            sub_dfall["stim_x"] = sub_dfall["stim_x"]*2
            sub_dfall["stim_y"] = sub_dfall["stim_y"]*2
            sub_dfall["stim_xdist"] = sub_dfall["stim_xdist"]*2
            sub_dfall["stim_ydist"] = sub_dfall["stim_ydist"]*2
            sub_dfall["training_axlocTL"] = sub_dfall["training_axlocTL"]*2
            sub_dfall["training_axlocTR"] = sub_dfall["training_axlocTR"]*2

            # trainingstim filter 
            navi_filter = sub_dfall.taskname.to_numpy() == "navigation"
            noncenter_filter = [np.logical_xor(x==0, y==0) for x,y in sub_dfall[["stim_x","stim_y"]].to_numpy()]
            training_nc_filter = np.vstack([navi_filter,noncenter_filter]).all(axis=0)
            curr_df, curr_X = sub_dfall[training_nc_filter].copy().reset_index(drop=True), navi_X[training_nc_filter,:]

            # split into sessions
            Xs, Ss = split_data(curr_X,curr_df.stim_session.values, return_groups=True)
            dfs = [curr_df[curr_df.stim_session==s].copy().reset_index(drop=True) for s in Ss]
            sub_patterns[roi].append(Xs)
            sub_stimdfs[roi].append(dfs)


    print("Computing PS and null distribution of PS from permutation")
    # place holders
    sub_cs_perms = {}
    sub_cs_obs = {}
    with Parallel(n_jobs=10) as parallel:
        for roi in rois:
            sub_cs_perms[roi] = [] # will be a nsub x n_perm x n_run  matrix that store the PS for permuted activity in each run for each participants
            sub_cs_obs[roi] = []   # will be a nsub x n_run matrix that store the PS for observed activity in each run for each participants
            for subXs,subdfs,subid in list(zip(sub_patterns[roi],sub_stimdfs[roi],subid_list)):
                print(f"{roi} - {subid}",end="\r",flush=True)
                stim_locs = [df[["stim_x","stim_y"]].to_numpy() for df in subdfs]
                uniqueaxlocs = [-2,-1,1,2]

                obs = [cal_PS(X,slocs,uniqueaxlocs) for X,slocs in zip(subXs,stim_locs)]
                # by specifying p as the random seed, we obtain the the same random shuffling for each ROI in each participants 
                permutations = parallel(delayed(compute_PSshuffled_for_allruns)(subXs,stim_locs,uniqueaxlocs,p) for p in range(n_perm))
                
                sub_cs_perms[roi].append(permutations)
                sub_cs_obs[roi].append(obs)
            dump({"permutation":sub_cs_perms[roi],"observation":sub_cs_obs[roi]},os.path.join(resoutput_dir,f"{runwisePS_savefn}_{roi}.pkl"))        
    
    dump({"permutation":sub_cs_perms,"observation":sub_cs_obs},os.path.join(resoutput_dir,f"{runwisePS_savefn}.pkl"))   


if run_classification:
    rois = ["allgtlocPrecentral_left","HPC_bilateral","V1_bilateral","vmPFC_bilateral","testgtlocParietalSup_bilateral"]
    
    ## We get the avergaed PS across runs and the corresponding null 
    averagePS, nullPS = {},{}
    for roi in rois:
        print(f"{roi}: \nLoading data for PS and null distribution of PS from permutation")
        loadedres = load(os.path.join(resoutput_dir,f"{runwisePS_savefn}_{roi}.pkl"))
        nulldists, obs = loadedres["permutation"], loadedres["observation"]

        print("Obtaining participant average PS and group null of PS")
        nullPS[roi] = [np.mean(sub_run_null,axis=1) for sub_run_null in nulldists] # nsub x nperm
        averagePS[roi] = np.mean(obs,axis=1)

        assert all([x.size == n_perm for x in nullPS[roi]])
        assert averagePS[roi].size == len(subid_list)

    dump({"permutation":nullPS,"observation":averagePS},os.path.join(resoutput_dir,f"{averagePS_savefn}.pkl"))   

    ## and classify participants 
    print("Classifying participants based on PS")
    ps_crit = 0.005
    subrep_category = {}
    for roi in averagePS.keys():
        subrep_category[roi] = []
        cs_null = np.mean(nullPS[roi],axis=0)
        for subps in averagePS[roi]:
            if np.mean(cs_null<subps)<ps_crit:
                subrep_category[roi].append(-1)
            elif np.mean(cs_null>subps)<ps_crit:
                subrep_category[roi].append(1)
            else:
                subrep_category[roi].append(0)

    PS_classification = pd.DataFrame(subrep_category)
    PS_classification["subid"] = subid_list
    PS_classification.to_csv(os.path.join(resoutput_dir,"PS_classification.csv"))

