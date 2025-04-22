"""
This file computes the dimensionality of feature representation for each axis using cross-validated SVD.
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
from scripts.Exp1_fmri.multivariate.pirateOMutils import cross_val_SVD

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
rois = list(roi_data.keys())

resoutput_dir = os.path.join(ROIRSAdir,"cvSVD_dimsionality")
checkdir(resoutput_dir)

## Run CV-SVD for training stimuli
print("\nRun CV-SVD for training stimuli\n")
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
            
            est_d, reconstruction_corr,reconstruction_r2,unique_sess = cross_val_SVD(whitenedX,session_labels=session_labels)
            #print(roi,subid,axisname, est_d)    
            tmpdf = pd.DataFrame()
            tmpdf["est_dim"] = est_d
            tmpdf["reconstruction_corr"] = reconstruction_corr
            tmpdf["reconstruction_r2"] = reconstruction_r2
            tmpdf["run"] = unique_sess
            tmpdf = tmpdf.assign(
                subid=subid,
                roi=roi,
                dimonaxis = axisname
            )

            svdestdim_dfs.append(tmpdf)

svdestdim_df = pd.concat(svdestdim_dfs,axis=0).reset_index(drop=True)
svdestdim_df.to_csv(os.path.join(resoutput_dir,"trainingstim_withinax_svdestdim_df.csv"),index=False)


## Run CV-SVD for test stimuli within axis
print("\nRun CV-SVD for test stimuli\n")
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
                
                est_d, reconstruction_corr,reconstruction_r2,unique_sess = cross_val_SVD(whitenedX,session_labels=session_labels)
                #print(roi,subid,axisname, est_d)    
                tmpdf = pd.DataFrame()
                tmpdf["est_dim"] = est_d
                tmpdf["reconstruction_corr"] = reconstruction_corr
                tmpdf["reconstruction_r2"] = reconstruction_r2
                tmpdf["run"] = unique_sess
                tmpdf = tmpdf.assign(
                    subid=subid,
                    roi=roi,
                    dimonaxis = axisname,
                    contorlaxisloc = t
                )

                testsvdestdim_dfs.append(tmpdf)

testsvdestdim_df = pd.concat(testsvdestdim_dfs,axis=0).reset_index(drop=True)
testsvdestdim_df.to_csv(os.path.join(resoutput_dir,"teststim_withinax_svdestdim_df.csv"),index=False)