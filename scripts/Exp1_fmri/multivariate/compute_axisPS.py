import itertools
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import json
from copy import deepcopy
import os
import time
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri
from zpyhelper.MVPA.preprocessors import split_data

project_path = r'E:\pirate_fmri\Analysis'
import sys
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim,generate_filters

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Load data
project_path  = r'E:\pirate_fmri\Analysis'
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
studydata_dir = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")
print(f"N_cohort1 = {len(cohort1ids)}")
print(f"N_cohort2 = {len(cohort2ids)}")

cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

base_rois = ["HPC","vmPFC","V1","V2"]
rois =  [f"{x}_bilateral" for x in base_rois]
ROIRSAdir = os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1')
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))

## select activity pattern of non-center training stimuli
sub_patterns = {}
sub_stimdfs = {}
for roi in rois:
    sub_patterns[roi], sub_stimdfs[roi] = [],[]
    for subid,subdata in zip(subid_list,roi_data[roi]):
        
        preprocedX = deepcopy(subdata["preprocX"])
        stimdf = subdata["stimdf"].copy()
        
        navi_filter = stimdf.taskname.to_numpy() == "navigation"

        #average across sessions, so we get a 25*nvoxel matrix
        navi_X = np.mean(split_data(X      = preprocedX[navi_filter,:],
                                    groups = stimdf[navi_filter].copy().stim_session.to_numpy()),
                        axis=0)
        
        #filter out the df for just one session
        sub_dfall = stimdf[navi_filter&(stimdf.stim_session==0)].copy().reset_index(drop=True)
        sub_dfall['training_axsetstr'] = ["x" if axs==0 else "y" for axs in sub_dfall['training_axset']]
        # make loc into integer
        sub_dfall["stim_x"] = sub_dfall["stim_x"]*2
        sub_dfall["stim_y"] = sub_dfall["stim_y"]*2
        sub_dfall["stim_xdist"] = sub_dfall["stim_xdist"]*2
        sub_dfall["stim_ydist"] = sub_dfall["stim_ydist"]*2
        sub_dfall["training_axlocTL"] = sub_dfall["training_axlocTL"]*2
        sub_dfall["training_axlocTR"] = sub_dfall["training_axlocTR"]*2

        #generate filters for various stimuli type
        curr_filt = generate_filters(sub_dfall)["training_nocenter"]
        curr_df, curr_X = sub_dfall[curr_filt].copy().reset_index(drop=True), navi_X[curr_filt,:]

        sub_patterns[roi].append(curr_X)
        sub_stimdfs[roi].append(curr_df)


## calculate PS and null distribution of PS from permutation 10000
# initialse random generator for reproducibility
rng = np.random.default_rng(123)
# define number of permutation
n_perm  = 10000
# place holders
sub_cs_perms = {}
sub_cs_obs = {}

def compute_shuffle(sX,sdf,locs,randgenseed=1): 
    randomgen = np.random.default_rng(randgenseed)
    pX = sX[randomgen.permutation(sX.shape[0]),:]                
    pxstims = np.array([pX[sdf.stim_x == j,:][0] for j in locs])
    pystims = np.array([pX[sdf.stim_y == j,:][0] for j in locs])
    perm_cossims = parallel_axes_cosine_sim(pxstims,pystims,return_codingdirs=False)
    return upper_tri(perm_cossims)[0].mean()

with Parallel(n_jobs=10) as parallel:
    for roi in rois:
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

dump({"permutation":sub_cs_perms,"observation":sub_cs_obs},os.path.join(ROIRSAdir,"ROI_PS_trainingnocenter_permnocentering.pkl"))        