"""
This file extracts data from different anatomical or functional ROIs
"""

import numpy as np
import json
import time
import glob
from copy import deepcopy
import os
from joblib import Parallel, delayed, cpu_count, dump,load

project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
roi_analysise_dir = os.path.join(fmridata_dir,'ROIdata')

import sys
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.mvpa_runner import MVPARunner


with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]


#ANATOMICAL ROIs
base_rois =  ["PPC","HPC","vmPFC","V1"]
rois = dict(zip([f"{x}_bilateral" for x in base_rois],[f"{x}_bilateral" for x in base_rois]))
save_file_name = "roi_data_4r"


########################################## EXTRACT DATA ##########################################
maskdir = roi_analysise_dir
beta_dir = {
    "navigation":[os.path.join(fmridata_dir,'unsmoothedLSA','LSA_stimuli_navigation')],
    "localizer": [os.path.join(fmridata_dir,'unsmoothedLSA','LSA_stimuli_localizer')],
}
beta_fname = {
    "navigation":['stimuli_4r.nii'],
    "localizer":['stimuli_1r.nii']
}
n_sess={
    "navigation":4,
    "localizer":1
}
cross_task_beta_dir   = sum(list(beta_dir.values()),[])
cross_task_beta_fname = sum(list(beta_fname.values()),[])
cross_task_vsmask_dir = cross_task_beta_dir
cross_task_vsmask_fname = ['mask.nii']*len(cross_task_vsmask_dir)
cross_task_res_dir   = beta_dir["navigation"]*n_sess["navigation"] + beta_dir["localizer"]*n_sess["localizer"]
cross_task_res_fname = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["navigation"])] + [f'resid_run{j+1}.nii.gz' for j in range(n_sess["localizer"])]

config_modelrdm_ = {"nan_identity":False, "splitgroup":True}
cross_task_beta_preproc = {# here we skip the averaging step because decoding analysis will need runspecific data
                           "preproc":{"MVNN": [None]*2 },
                           "distance_metric":"correlation"}

data = {}
with Parallel(n_jobs=15) as parallel:
    for roi,roifn in rois.items():
        print(f"extracting ROI data from {roi}")
        data[roi] = []
        CrossTaskRSA = MVPARunner(
                        participants=subid_list, fmribeh_dir=fmribeh_dir,
                        beta_dir   = cross_task_beta_dir,   beta_fname   = cross_task_beta_fname,
                        vsmask_dir = cross_task_vsmask_dir, vsmask_fname = cross_task_vsmask_fname,
                        pmask_dir  = cross_task_vsmask_dir,  pmask_fname  = cross_task_vsmask_fname,
                        res_dir    = cross_task_res_dir, 
                        res_fname = cross_task_res_fname,
                        anatmasks = [os.path.join(maskdir,f"{roifn}.nii")],
                        taskname  = "both",
                        config_modelrdm  = config_modelrdm_,
                        config_neuralrdm = cross_task_beta_preproc)
        outputs =  parallel(delayed(CrossTaskRSA.get_neuralRDM)(subid) for subid in subid_list)
        for op,subid in zip(outputs,subid_list):
            print(f"      organizing data for {subid}", end="\r", flush=True)
            preprocX, _, rawX  = op
            stimdf             = CrossTaskRSA.get_modelRDM(subid=subid).stimdf.copy()
            stimdf["taskname"] = ["localizer" if s==np.max(stimdf["stim_session"]) else "navigation" for s in np.array(stimdf["stim_session"])]
            
            subdata = {
                "preprocX":preprocX,
                "rawX":rawX,
                "stimdf":stimdf
            }
            data[roi].append(subdata)
        dump(data[roi],os.path.join(roi_analysise_dir,f"{save_file_name}_{roi}.pkl"))

# save all data
data={}
for roi,roifn in rois.items():
    data[roi] = load(os.path.join(roi_analysise_dir,f"{save_file_name}_{roi}.pkl"))

dump(data,os.path.join(roi_analysise_dir,f"{save_file_name}.pkl"))