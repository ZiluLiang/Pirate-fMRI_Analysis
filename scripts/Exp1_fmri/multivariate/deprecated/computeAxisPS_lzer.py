"""
This file computes the PS and null distribution of PS for the training locations that are not the center location.
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

# Load data
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
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois = [x for x in list(roi_data.keys()) if "bilateral" in x] + ["allgtlocPrecentral_left"]

resoutput_dir = os.path.join(ROIRSAdir,"axisPS")
checkdir(resoutput_dir)

## select activity pattern of non-center training stimuli
sub_patterns = {}
sub_stimdfs = {}
for ir, roi in enumerate(rois):
    print(f"{ir+1}/{len(rois)}",end="\n")
    sub_patterns[roi], sub_stimdfs[roi] = [],[]
    for subid,subdata in zip(subid_list,roi_data[roi]):
        
        preprocedX = deepcopy(subdata["preprocX"])
        stimdf = subdata["stimdf"].copy()
        
        # localizer only has one run, so we don't need to average across session
        lzer_filter = stimdf.taskname.to_numpy() == "localizer"
        noncenter_filter = [np.logical_xor(x==0, y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]
        lzer_nc_filter = np.vstack([lzer_filter,noncenter_filter]).all(axis=0)
        curr_X = preprocedX[lzer_nc_filter,:]
        curr_df = stimdf[lzer_nc_filter].copy().reset_index(drop=True)
        
        # make loc into integer
        curr_df["stim_x"] = curr_df["stim_x"]*2
        curr_df["stim_y"] = curr_df["stim_y"]*2
        curr_df["stim_xdist"] = curr_df["stim_xdist"]*2
        curr_df["stim_ydist"] = curr_df["stim_ydist"]*2
        curr_df["training_axlocTL"] = curr_df["training_axlocTL"]*2
        curr_df["training_axlocTR"] = curr_df["training_axlocTR"]*2

        sub_patterns[roi].append(curr_X)
        sub_stimdfs[roi].append(curr_df)

def compute_shuffle(sX,sdf,locs,randgenseed=1): 
    # initialse random generator for reproducibility
    randomgen = np.random.default_rng(randgenseed)
    pX = sX[randomgen.permutation(sX.shape[0]),:]                
    pxstims = np.array([pX[sdf.stim_x == j,:][0] for j in locs])
    pystims = np.array([pX[sdf.stim_y == j,:][0] for j in locs])
    perm_cossims = parallel_axes_cosine_sim(pxstims,pystims,return_codingdirs=False)
    return upper_tri(perm_cossims)[0].mean()

## calculate PS and null distribution of PS from permutation 10000
# define number of permutation
n_perm  = 10000
# place holders
sub_cs_perms = {}
sub_cs_obs = {}
with Parallel(n_jobs=10) as parallel:
    for ir,roi in enumerate(rois):
        print(f"\nComputing PS in {ir+1}/{len(rois)}",end="\n")

        sub_cs_perms[roi] = []
        sub_cs_obs[roi] = []
        for subX,subdf,subid in list(zip(sub_patterns[roi],sub_stimdfs[roi],subid_list)):
            print(f"{roi} - {subid}",end="\r",flush=True)
            axlocs = np.unique(subdf.training_axlocTL)

            assert all([np.sum(subdf.stim_x == j)==1 for j in axlocs])
            assert all([np.sum(subdf.stim_y == j)==1 for j in axlocs])
            # pick x/y stims and put them into same order
            xstims = np.array([subX[subdf.stim_x == j,:][0] for j in axlocs])
            ystims = np.array([subX[subdf.stim_y == j,:][0] for j in axlocs])
            
            obs = upper_tri(parallel_axes_cosine_sim(xstims,ystims))[0].mean()
            # by specifying p as the random seed, we obtain the the same random shuffling for each ROI in each participants 
            permutations = parallel(delayed(compute_shuffle)(subX,subdf,axlocs,p) for p in range(n_perm))
            
            sub_cs_perms[roi].append(permutations)
            sub_cs_obs[roi].append(obs)

dump({"permutation":sub_cs_perms,"observation":sub_cs_obs},os.path.join(resoutput_dir,"aveactivitypatPS_NCtrainingloc.pkl"))        