"""
This file extracts data from different anatomical or functional ROIgit add
"""

import numpy as np
import json
import time
import glob
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump,load

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'src'))
from zpyhelper.filesys import checkdir

from multivariate.rsa_runner import RSARunner


project_path = r'E:\pirate_fmri\Analysis'
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
outputdata_dir  = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]

# #FUNCTIONAL ROI: mvnn_wbsearch_reg_compete_featurecartesian_combinexy_testpairs_between
roi_analysise_dir = os.path.join(fmridata_dir,'ROIRSA','mvnn_wbsearch_reg_compete_featurecartesian_combinexy_withsg_between')
rois = ["stimuligroup_FrontalMid2L","stimuligroup_FrontalSup2L","stimuligroup_ParietalInfL","stimuligroup_SMAL"] + \
       ["sphere_neggtloc_FrontalMid2L",
        "sphere_posgtloc_CalcarineL",
        "sphere_posgtloc_CalcarineR",
        "sphere_posgtloc_HippocampusR",
        "sphere_posgtloc_PrecentralL"]
# #FUNCTIONAL ROI: mvnn_wbsearch_reg_compete_featurecartesian_combinexy_testpairs_between
roi_analysise_dir = os.path.join(fmridata_dir,'ROIRSA','mvnn_wbsearch_reg_compete_featurecartesian_combinexy_testpairs_between')
rois = [#"cluster_posgtlocGnG_CuneusPrecuneusR",
        #"cluster_posgtlocGnG_OccipitalMidParietalSupL",
        #"sphere_posgtlocGnG_ParaHippocampalR",
        "cluster_posgtlocG_CalcarineLingualRL","cluster_posgtlocG_PostcentralParietalSupL",
        #"cluster_posfeatureG_OccipitalInfRL",
        "cluster_posfeatureG_ACCpreFrontalMedOrbL","cluster_posfeatureG_FrontalInfOperFrontalMid2L","cluster_posfeatureG_PostcentralR",
        "cluster_posfeaturenolocG_occipital"]
# #ANATOMICAL ROI
# roi_analysise_dir = os.path.join(fmridata_dir,'ROIRSA','anatomical')
# rois =  ["frontal_bilateral","ofc_bilateral", "parietal_bilateral",  "hippocampus_bilateral","parahippocampus_bilateral"] + \
#         ["rV1", "rV2", "rV3dva", "rV4v", "rV5", "rV6"]
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
cross_task_beta_preproc = {"preproc":{"MVNN": [None]*2#,"ATOE": [None]*2
                                      }, 
                           "distance_metric":"correlation"}

data = {}
with Parallel(n_jobs=10) as parallel:
    for roi in rois:
        print(roi)
        data[roi] = []
        CrossTaskRSA = RSARunner(
                        participants=subid_list, fmribeh_dir=fmribeh_dir,
                        beta_dir   = cross_task_beta_dir,   beta_fname   = cross_task_beta_fname,
                        vsmask_dir = cross_task_vsmask_dir, vsmask_fname = cross_task_vsmask_fname,
                        pmask_dir  = cross_task_vsmask_dir,  pmask_fname  = cross_task_vsmask_fname,
                        res_dir    = cross_task_res_dir, 
                        res_fname = cross_task_res_fname,
                        anatmasks = [os.path.join(maskdir,f"{roi}.nii")],
                        taskname  = "both",
                        config_modelrdm  = config_modelrdm_,
                        config_neuralrdm = cross_task_beta_preproc)
        outputs =  parallel(delayed(CrossTaskRSA.get_neuralRDM)(subid) for subid in subid_list)
        for op,subid in zip(outputs,subid_list):
            preprocX, _, rawX  = op
            stimdf             = CrossTaskRSA.get_modelRDM(subid=subid).stimdf.copy()
            stimdf["taskname"] = ["localizer" if s==np.max(stimdf["stim_session"]) else "navigation" for s in np.array(stimdf["stim_session"])]
            
            subdata = {
                "preprocX":preprocX,
                "rawX":rawX,
                "stimdf":stimdf
            }
            data[roi].append(subdata)

dump(data,os.path.join(roi_analysise_dir,"feature_gtloc_roi_data_4r.pkl"))