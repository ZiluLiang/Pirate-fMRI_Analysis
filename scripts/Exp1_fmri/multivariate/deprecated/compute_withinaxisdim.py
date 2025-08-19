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

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import cross_val_SVD

import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,r2_score, confusion_matrix

import scipy

import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

## Load data
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

ROIRSAdir = os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1andFUNCcluster')
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois =  list(roi_data.keys())

cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))


## Run CV-SVD for training stimuli
svdestdim_dfs = []
for roi in rois:
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)

        navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
        training_filter = subdata["stimdf"].stim_group.to_numpy() == 1
        non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]

        filter_x = subdata["stimdf"].stim_y.to_numpy() == 0
        filter_y = subdata["stimdf"].stim_x.to_numpy() == 0

        for axisname, axisfil in zip(["x axis","y axis"],[filter_x,filter_y]):
            fil = np.vstack([navi_filter,training_filter,non_center_filter,axisfil]).all(axis=0)

            whitenedX = subdata["preprocX"][fil]
            session_labels = subdata["stimdf"][fil].stim_session.to_numpy()
            
            est_d, reconstruction_corr,unique_sess = cross_val_SVD(whitenedX,session_labels=session_labels)
            #print(roi,subid,axisname, est_d)    
            tmpdf = pd.DataFrame()
            tmpdf["est_dim"] = est_d
            tmpdf["reconstruction_corr"] = reconstruction_corr
            tmpdf["run"] = unique_sess
            tmpdf = tmpdf.assign(
                subid=subid,
                roi=roi,
                dimonaxis = axisname
            )

            svdestdim_dfs.append(tmpdf)

svdestdim_df = pd.concat(svdestdim_dfs,axis=0).reset_index(drop=True)
svdestdim_df.to_csv(os.path.join(ROIRSAdir,"trainingstim_withinax_svdestdim_df.csv"),index=False)


## Run CV-SVD for test stimuli within axis
testsvdestdim_dfs = []
for roi in rois:
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)

        navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
        test_filter = subdata["stimdf"].stim_group.to_numpy() == 0

        tlocs = subdata["stimdf"][test_filter].copy().stim_x.unique()
        
        for t in tlocs:
            filter_x = subdata["stimdf"].stim_y.to_numpy() == t
            filter_y = subdata["stimdf"].stim_x.to_numpy() == t

            for axisname, axisfil in zip(["x axis","y axis"],[filter_x,filter_y]):
                fil = np.vstack([navi_filter,test_filter,axisfil]).all(axis=0)

                whitenedX = subdata["preprocX"][fil]
                session_labels = subdata["stimdf"][fil].stim_session.to_numpy()
                
                est_d, reconstruction_corr,unique_sess = cross_val_SVD(whitenedX,session_labels=session_labels)
                #print(roi,subid,axisname, est_d)    
                tmpdf = pd.DataFrame()
                tmpdf["est_dim"] = est_d
                tmpdf["reconstruction_corr"] = reconstruction_corr
                tmpdf["run"] = unique_sess
                tmpdf = tmpdf.assign(
                    subid=subid,
                    roi=roi,
                    dimonaxis = axisname,
                    contorlaxisloc = t
                )

                testsvdestdim_dfs.append(tmpdf)

testsvdestdim_df = pd.concat(testsvdestdim_dfs,axis=0).reset_index(drop=True)
testsvdestdim_df.to_csv(os.path.join(ROIRSAdir,"teststim_withinax_svdestdim_df.csv"),index=False)