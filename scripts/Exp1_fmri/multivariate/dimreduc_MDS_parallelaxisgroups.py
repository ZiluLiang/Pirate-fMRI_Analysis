"""
This script performs MDS analysis on the non-centre training stimuli RDMs obtained from the fMRI data of the pirate experiment.
It performs the MDS for different parallel axis groups
"""
import itertools
import numpy as np
import seaborn as sns

import json
import time
import pandas as pd
import glob
from copy import deepcopy
import os
import sys
from joblib import dump,load

from zpyhelper.MVPA.preprocessors import  split_data, concat_data
from zpyhelper.MVPA.rdm import compute_rdm
from zpyhelper.filesys import checkdir


import matlab.engine

import scipy

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')

sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import  minmax_scale, generate_filters


with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]

print("N_cohort 1: ",len(cohort1ids), "  N_cohort 2: ",len(cohort2ids), "N_Total: ",len(subid_list))
cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))

respar_dir = os.path.join(ROIRSAdir,"axisPS")
# first we load the classification results
PS_classification = pd.read_csv(os.path.join(respar_dir,"PS_classification.csv"))

# set directories
resoutput_dir = os.path.join(respar_dir,"MDS")
checkdir(resoutput_dir)

# set ROIs
rois = ["HPC_bilateral","V1_bilateral","vmPFC_bilateral","PPC_bilateral"]
classifyby = "HPC_bilateral"
curr_classifications = PS_classification[classifyby].values
PSgroupnamestr = {1:"top-left",-1:"top-right",0:"orthogonal"}
PSgroups = list(PSgroupnamestr.keys())

def get_activity_pattern_for_stimsubset(subdata,stimsubset):
    preprocedX = subdata["preprocX"]
    stimdf = subdata["stimdf"].copy()
    
    #average across sessions, so we get a 25*nvoxel matrix
    task_Xs, tasks = split_data(X      = preprocedX,
                        groups = stimdf.taskname.to_numpy(),
                        select_groups=["navigation","localizer"], # make sure it is in the right order
                        return_groups = True)
    mean_task_Xs = concat_data([np.mean(split_data(x,stimdf[stimdf.taskname==t].copy().stim_session.values),axis=0) for t,x in zip(tasks,task_Xs)])
    
    #filter out the df for just one session
    sub_dfall = stimdf[stimdf.stim_session.isin([0,4])].copy().reset_index(drop=True)
    sub_dfall['training_axsetstr'] = ["x" if axs==0 else "y" if axs==1 else "center" if np.logical_and(sx==0,sy==0) else "na" for axs,sx,sy in sub_dfall[['training_axset','stim_x','stim_y']].to_numpy()]
    # make loc into integer
    sub_dfall["stim_x"] = sub_dfall["stim_x"]*2
    sub_dfall["stim_y"] = sub_dfall["stim_y"]*2
    sub_dfall["stim_xdist"] = sub_dfall["stim_xdist"]*2
    sub_dfall["stim_ydist"] = sub_dfall["stim_ydist"]*2
    sub_dfall["training_axlocTL"] = sub_dfall["training_axlocTL"]*2
    sub_dfall["training_axlocTR"] = sub_dfall["training_axlocTR"]*2

    #generate filters for various stimuli type
    subfilters = generate_filters(sub_dfall)
    if isinstance(stimsubset,str):
        assert stimsubset in subfilters.keys(), f"stimsubset {stimsubset} not in [{subfilters.keys()}]"        

        curr_filt = subfilters[stimsubset]
        curr_df, curr_X = sub_dfall[curr_filt].copy().reset_index(drop=True), mean_task_Xs[curr_filt,:]
        

        return curr_X,curr_df
    
    elif isinstance(stimsubset,list):
        assert [x in subfilters.keys() for x in stimsubset], f"stimsubset: {[x for x in stimsubset if x not in subfilters.keys()]} not in [{subfilters.keys()}]"

        Xs,dfs = {},{}
        for subset in stimsubset:
            curr_filt = subfilters[subset]
            curr_df, curr_X = sub_dfall[curr_filt].copy().reset_index(drop=True), mean_task_Xs[curr_filt,:]
            Xs[subset],dfs[subset] = curr_X,curr_df

        return Xs,dfs
    else:
        raise ValueError("stimsubset must be either a string or a list of strings")

def sort_activitypattern_with_stimdf(X,stimdf,sortby):
    if isinstance(sortby,str):
        assert sortby in stimdf.columns, f"sortby {sortby} not in [{stimdf.columns}]"
    elif isinstance(sortby,list):
        assert [x in stimdf.columns for x in sortby], f"sortby: {[x for x in sortby if x not in stimdf.columns]} not in [{stimdf.columns}]"
    else:    
        raise ValueError("must specify sortby as a column name or a list of column names in stimdf")

    stimdf = stimdf.copy().reset_index(drop=True)

    neworder_df = stimdf.sort_values(by=sortby,inplace=False)

    return X[neworder_df.index,:],neworder_df

# then we get data for each stimulisubset we want to analyse
sort_by_ccols = {
    "allstim":           ["stim_group","training_axsetstr","training_axlocTL","stim_x","stim_y"],
    "allstim_nocenter":  ["stim_group","training_axsetstr","training_axlocTL","stim_x","stim_y"],
    "test":              ["stim_x","stim_y"],
    "training_all":      ["training_axsetstr","training_axlocTL"],
    "training_nocenter": ["training_axsetstr","training_axlocTL"],
    "trainloc_all":      ["training_axsetstr","training_axlocTL"],
    "trainloc_nocenter": ["training_axsetstr","training_axlocTL"] 
    }


subXs, subdfs = {"G":{}, "nG":{}}, {"G":{}, "nG":{}}
# then we obtain the data and rdm for each group
print("\nObtaining data and RDMs\n",end="\n")
sub_rdms = {}
stimdfs  = {}
for s, ssname in enumerate(sort_by_ccols.keys()):
    sub_rdms[ssname] = {}

    for ir,roi in enumerate(rois):
        print(f"{s+1}/{len(sort_by_ccols)} subsets  {ir+1}/{len(rois)} rois ",end="\r")
        sub_rdms[ssname][roi] = {}
        
        for psgroup_num, psgroup_str in PSgroupnamestr.items():       
            Gdata  = [sort_activitypattern_with_stimdf(*get_activity_pattern_for_stimsubset(subdata,ssname), sortby=sort_by_ccols[ssname]) for subid,subdata,subpsgroup in zip(subid_list,roi_data[roi],curr_classifications) if np.logical_and(subid in generalizers, subpsgroup==psgroup_num)]
            nGdata = [sort_activitypattern_with_stimdf(*get_activity_pattern_for_stimsubset(subdata,ssname), sortby=sort_by_ccols[ssname]) for subid,subdata,subpsgroup in zip(subid_list,roi_data[roi],curr_classifications) if np.logical_and(subid in nongeneralizers, subpsgroup==psgroup_num)]

            sub_rdms[ssname][roi][psgroup_str] = {}            
            sub_rdms[ssname][roi][psgroup_str]["G"]  = [compute_rdm(subd[0],"correlation") for subd in Gdata]
            sub_rdms[ssname][roi][psgroup_str]["nG"] = [compute_rdm(subd[0],"correlation") for subd in nGdata]
    stimdfs[ssname] = Gdata[0][1]        
            
# Then we compute the MDS
print("\nComputing MDS\n",end="\n")
matlab_eng = matlab.engine.start_matlab()
average_rdms = {}
mds_results = {"2D":{},"3D":{}}
for s, ssname in enumerate(sort_by_ccols.keys()):
    average_rdms[ssname] = {}
    mds_results["2D"][ssname] = {}
    mds_results["3D"][ssname] = {}
    for ir,roi in enumerate(rois):
        print(f"{s+1}/{len(sort_by_ccols)} subsets  {ir+1}/{len(rois)} rois ",end="\r")
        average_rdms[ssname][roi] = {}
        mds_results["2D"][ssname][roi] = {}
        mds_results["3D"][ssname][roi] = {}
        for psgroup_num, psgroup_str in PSgroupnamestr.items():   
            average_rdms[ssname][roi][psgroup_str] = {}
            mds_results["2D"][ssname][roi][psgroup_str] = {} 
            mds_results["3D"][ssname][roi][psgroup_str] = {}            
            
            average_rdms[ssname][roi][psgroup_str]["G"]  = np.mean([minmax_scale(m,newmin=0,newmax=1) for m in sub_rdms[ssname][roi][psgroup_str]["G"]],axis=0)
            average_rdms[ssname][roi][psgroup_str]["nG"] = np.mean([minmax_scale(m,newmin=0,newmax=1) for m in sub_rdms[ssname][roi][psgroup_str]["nG"]],axis=0)

            for sg in ["G","nG"]:
                for ncompo in [2,3]:
                    stimdf = stimdfs[ssname].copy()
                    ave_rdm = minmax_scale(average_rdms[ssname][roi][psgroup_str][sg],newmin=0,newmax=1)
                    
                    #mdsres = np.array(matlab_eng.mdscale(ave_rdm,matlab_eng.double(ncompo),
                    #                                     'Options',matlab_eng.struct('MaxIter',matlab.double(10000))
                    #                                     )
                    #                     )
                    mdsres = np.array(matlab_eng.cmdscale(ave_rdm,matlab_eng.double(ncompo))
                                        )
                    stimdf[[f"MDS{i+1}" for i in range(ncompo)]] = mdsres
                    mds_results[f"{ncompo}D"][ssname][roi][psgroup_str][sg] = stimdf

matlab_eng.quit()
dump({"aveRDM":average_rdms,"MDSres":mds_results},os.path.join(resoutput_dir,"ParallAx_nonmetricmds_results.pkl"))