import itertools
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

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import compute_rdm,lower_tri,upper_tri, compute_rdm_nomial, compute_rdm_identity
from zpyhelper.MVPA.preprocessors import scale_feature, average_odd_even_session,normalise_multivariate_noise, split_data, concat_data,extract_pc,average_flexi_session
from zpyhelper.MVPA.estimators import PatternCorrelation, MultipleRDMRegression, NeuralRDMStability

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM
from scripts.Exp1_fmri.multivariate.mvpa_runner import MVPARunner
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim, minmax_scale, generate_filters
from src.utils.composition_modelfit import multi_start_optimisation

import sklearn
from sklearn.manifold import MDS,TSNE 
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,r2_score, confusion_matrix

import scipy

import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore', category=FutureWarning)


study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
studydata_dir  = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]

print("N_cohort 1: ",len(cohort1ids), "  N_cohort 2: ",len(cohort2ids), "N_Total: ",len(subid_list))

fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
fmribeh_dir = os.path.join(fmridata_dir,'beh')

base_rois = ["HPC","vmPFC","V1","V2"]
anatrois =  [f"{x}_bilateral" for x in base_rois]
anatROIRSAdir = os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1')
anatroi_data = load(os.path.join(anatROIRSAdir,"roi_data_4r.pkl"))
rois = anatrois#[:2]
roi_data = anatroi_data
ROIRSAdir = anatROIRSAdir

cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

preprox_fun = lambda x,sess: extract_pc(scale_feature(x,1))#scale_feature(x,1)#concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #extract_pc(scale_feature(x,1)) #
baseclf_kwargs = {'max_iter':-1,'kernel':'linear'}
gridsearcg_paramgrid={'C':np.linspace(10**-10,10,num=50)} #'tol':np.logspace(-5,-3,num=5),
baseclf_kwargs = {'max_iter':1000000,'solver':'lbfgs','multi_class':'multinomial'}
gridsearcg_paramgrid={'C':np.linspace(10**-10,10,num=50)} #'tol':np.logspace(-5,-3,num=3),
        
res_dfs = []
confusion_mats = {}
for roi in rois:
    confusion_mats[roi] = {}
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)
        confusion_mats[roi][subid] = {}

        navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
        training_filter = subdata["stimdf"].stim_group.to_numpy() == 1
        non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
        filter_x = subdata["stimdf"].stim_y.to_numpy() == 0
        filter_y = subdata["stimdf"].stim_x.to_numpy() == 0
        
        
        for tarvar in ["stim_x","stim_y","stim_id"]:
            confusion_mats[roi][subid][tarvar] = []
            if tarvar=="stim_x":
                fil = np.vstack([navi_filter,training_filter,non_center_filter,filter_x]).all(axis=0)
            elif tarvar=="stim_y":
                fil = np.vstack([navi_filter,training_filter,non_center_filter,filter_y]).all(axis=0)
            else:
                fil = np.vstack([navi_filter,training_filter,non_center_filter]).all(axis=0)

            whitenedX = subdata["preprocX"][fil]
            session_labels = subdata["stimdf"][fil].stim_session.to_numpy()
            preproced_X = preprox_fun(whitenedX,session_labels)
        
            
            target_labels = subdata["stimdf"][tarvar].to_numpy()[fil]
            if tarvar in ["stim_x","stim_y"]:
                target_labels = (target_labels*2+2).astype(int)

            fit_accs, eval_accs = [],[]
            for heldout_sessions in np.unique(session_labels):
                
                fit_session_labels = session_labels[session_labels!=heldout_sessions]
                fit_X, fit_target = preproced_X[session_labels!=heldout_sessions], target_labels[session_labels!=heldout_sessions] 
                eval_X, eval_target = preproced_X[session_labels==heldout_sessions], target_labels[session_labels==heldout_sessions]
                
                
                clf = GridSearchCV(LogisticRegression(**baseclf_kwargs), # or SVC(**baseclf_kwargs),
                                   param_grid=gridsearcg_paramgrid,
                                   cv=LeaveOneGroupOut(),#RepeatedStratifiedKFold(n_splits=3),
                                   n_jobs=15)
                clf.fit(fit_X,fit_target,groups=fit_session_labels) #,groups=fit_session_labels

                fit_acc = clf.score(fit_X,fit_target)
                eval_acc = clf.score(eval_X,eval_target)
                eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                confusion_mats[roi][subid][tarvar].append(eval_cfm)
                fit_accs.append(fit_acc)
                eval_accs.append(eval_acc)
            tmpdf = pd.DataFrame()
            tmpdf["fit_acc"] = fit_accs
            tmpdf["eval_acc"] = eval_accs
            res_dfs.append(
                tmpdf.assign(roi=roi,
                         subid=subid,
                         target=tarvar)
                         )
res_df = pd.concat(res_dfs).reset_index(drop=True)
decoder = {"estimator":clf.estimator.__str__(),
           "gsparamgrid":gridsearcg_paramgrid,
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
      "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(fmridata_dir,"noncenter_trainingstim_LRdecoding_acc.pkl"))



##########################################################################################################
res_dfs = []
confusion_mats = {}
for roi in rois:
    confusion_mats[roi] = {}
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)
        confusion_mats[roi][subid] = {}

        navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
        training_filter = subdata["stimdf"].stim_group.to_numpy() == 1
        non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
        
        for tarvar in ["stim_group","stim_id"]:
            confusion_mats[roi][subid][tarvar] = []
            fil = navi_filter

            whitenedX = subdata["preprocX"][fil]
            session_labels = subdata["stimdf"][fil].stim_session.to_numpy()
            preproced_X = preprox_fun(whitenedX,session_labels)
        
            
            target_labels = subdata["stimdf"][tarvar].to_numpy()[fil]
            if tarvar in ["stim_x","stim_y"]:
                target_labels = (target_labels*2+2).astype(int)

            fit_accs, eval_accs = [],[]
            for heldout_sessions in np.unique(session_labels):
                
                fit_session_labels = session_labels[session_labels!=heldout_sessions]
                fit_X, fit_target = preproced_X[session_labels!=heldout_sessions], target_labels[session_labels!=heldout_sessions] 
                eval_X, eval_target = preproced_X[session_labels==heldout_sessions], target_labels[session_labels==heldout_sessions]
                
                clf = GridSearchCV(LogisticRegression(**baseclf_kwargs), # or SVC(**baseclf_kwargs),
                                    param_grid=gridsearcg_paramgrid,
                                    cv=LeaveOneGroupOut(),
                                    n_jobs=10)
                clf.fit(fit_X,fit_target,groups=fit_session_labels)

                fit_acc = clf.score(fit_X,fit_target)
                eval_acc = clf.score(eval_X,eval_target)
                eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                confusion_mats[roi][subid][tarvar].append(eval_cfm)
                fit_accs.append(fit_acc)
                eval_accs.append(eval_acc)
            tmpdf = pd.DataFrame()
            tmpdf["fit_acc"] = fit_accs
            tmpdf["eval_acc"] = eval_accs
            res_dfs.append(
                tmpdf.assign(roi=roi,
                         subid=subid,
                         target=tarvar)
                         )
res_df = pd.concat(res_dfs).reset_index(drop=True)
decoder = {"estimator":clf.estimator.__str__(),
           "gsparamgrid":gridsearcg_paramgrid,
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
      "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(fmridata_dir,"stimgroup_LRdecoding_acc.pkl"))