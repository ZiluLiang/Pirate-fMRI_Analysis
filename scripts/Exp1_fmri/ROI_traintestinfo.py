import itertools
import matlab.engine
import plotly.express as px

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
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_runner import RSARunner

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

project_path = r'E:\pirate_fmri\Analysis'
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
outputdata_dir  = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]

# rois = ["stimuligroup_FrontalMid2L","stimuligroup_FrontalSup2L","stimuligroup_ParietalInfL","stimuligroup_SMAL"]
# rois = ["sphere_neggtloc_FrontalMid2L",
#         "sphere_posgtloc_CalcarineL",
#         "sphere_posgtloc_CalcarineR",
#         "sphere_posgtloc_HippocampusR",
#         "sphere_posgtloc_PrecentralL"]
rois = ["cluster_posgtlocGnG_CuneusPrecuneusR","cluster_posgtlocGnG_OccipitalMidParietalSupL",
        "sphere_posgtlocGnG_ParaHippocampalR","cluster_posgtlocG_CalcarineLingualRL","cluster_posgtlocG_PostcentralParietalSupL"]
#roi_analysise_dir = os.path.join(fmridata_dir,'ROIRSA','mvnn_wbsearch_reg_compete_featurecartesian_combinexy_withsg_between')
roi_analysise_dir = os.path.join(fmridata_dir,'ROIRSA','mvnn_wbsearch_reg_compete_featurecartesian_combinexy_testpairs_between')
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


config_modelrdm_ = {"nan_identity":False, "splitgroup":True}
navi_mvnn_aoe = {"preproc":{"MVNN": [normalise_multivariate_noise,{}],
            "AOE": [average_odd_even_session,{}]}, 
            "distance_metric":"correlation"}

lzer_mvnn = {"preproc":{"MVNN": [normalise_multivariate_noise,{}]}, 
            "distance_metric":"correlation"}

data = {}
for roi in rois:
    roi_fn=f"{roi}.nii"
    data[roi] = []
    NavigationRSA = RSARunner(
                    participants=subid_list, fmribeh_dir=fmribeh_dir,
                    beta_dir   = beta_dir["navigation"],  beta_fname   = beta_fname["navigation"],
                    vsmask_dir = beta_dir["navigation"],  vsmask_fname = ['mask.nii']*len(beta_dir["navigation"]),
                    pmask_dir  = beta_dir["navigation"],  pmask_fname  = ['mask.nii']*len(beta_dir["navigation"]),
                    res_dir    = beta_dir["navigation"]*n_sess["navigation"], 
                    res_fname = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["navigation"])],
                    anatmasks =[os.path.join(maskdir,roi_fn)],
                    nsession  = n_sess["navigation"],
                    taskname  = "navigation",
                    config_modelrdm  = config_modelrdm_,
                    config_neuralrdm = navi_mvnn_aoe)
    
    LocalizerRSA = RSARunner(
                    participants=subid_list, fmribeh_dir=fmribeh_dir,
                    beta_dir   = beta_dir["localizer"], beta_fname   = beta_fname["localizer"],
                    vsmask_dir = beta_dir["localizer"], vsmask_fname = ['mask.nii']*len(beta_dir["localizer"]),
                    pmask_dir  = beta_dir["localizer"], pmask_fname  = ['mask.nii']*len(beta_dir["localizer"]),
                    res_dir    = beta_dir["localizer"]*n_sess["localizer"],
                    res_fname  = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["localizer"])],

                    anatmasks=[os.path.join(maskdir,roi_fn)],
                    nsession=n_sess["localizer"],
                    taskname="localizer",
                    config_modelrdm  = config_modelrdm_,
                    config_neuralrdm = lzer_mvnn)
    
    for subid in subid_list:
        print(f"retrieving {subid} data")
        print(f"navigation in {roi}", end="\r", flush=True)
        navi_preprocX, _, navi_rawX = NavigationRSA.get_neuralRDM(subid=subid)
        navi_stimdf                 = NavigationRSA.get_modelRDM(subid=subid).stimdf.copy()
        
        print(f"localizer in {roi}", end="\r", flush=True)
        lzer_preprocX, _, lzer_rawX = LocalizerRSA.get_neuralRDM(subid=subid)
        lzer_stimdf                 = LocalizerRSA.get_modelRDM(subid=subid).stimdf.copy()
        subdata = {
            "navigation-preprocX":navi_preprocX,
            "navigation-rawX":navi_rawX,
            "navigation-stimdf":navi_stimdf,
            "localizer-preprocX":lzer_preprocX,
            "localizer-rawX":lzer_rawX,
            "localizer-stimdf":lzer_stimdf
        }
        data[roi].append(subdata)

dump(data,os.path.join(roi_analysise_dir,"gtloc_roi_data.pkl"))