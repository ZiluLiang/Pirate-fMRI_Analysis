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
from zpyhelper.MVPA.preprocessors import split_data, scale_feature

import sys
project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')

sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import cross_val_SVD

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Load data
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois = list(roi_data.keys())

resoutput_dir = os.path.join(ROIRSAdir,"cvSVD_dimsionality")
checkdir(resoutput_dir)

## Run CV-SVD for training stimuli
print("\nRun CV-SVD for training stimuli\n")
svdestdim_dfs = []
svdprojection_dfs = []
meanXsvdprojection_dfs = []
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

            #make sure the order is correct:
            curr_df = subdata["stimdf"][fil].copy().reset_index().sort_values(by=["stim_session","training_axlocTL"])
            row_order = curr_df.index.to_numpy()
            whitenedX = whitenedX[row_order,:]

            # get session labels and features
            session_labels = curr_df.stim_session.to_numpy()
            stim_features = curr_df["training_axlocTL"].to_numpy()

            est_d, reconstruction_corr,reconstruction_r2,unique_sess,projection_1D = cross_val_SVD(whitenedX,session_labels=session_labels)
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
            
            #compute a projection of 1D for all data
            mean_X = scale_feature(np.mean(split_data(X=whitenedX,groups=session_labels),axis=0),1,standardize=False)
            u,s,vh = np.linalg.svd(mean_X,full_matrices=False)
            reconst1D_meanX = np.dot(u[:,:1],s[:1])            

            meanXsvdprojection_dfs.append(
                pd.DataFrame(
                    {
                        "SVD_projection": reconst1D_meanX,
                        "axisloc": stim_features[session_labels==0],
                    }).assign(
                        #run="averaged",
                        subid=subid,
                        roi=roi,
                        dimonaxis = axisname)
                    )

svdestdim_df = pd.concat(svdestdim_dfs,axis=0).reset_index(drop=True)
svdestdim_df.to_csv(os.path.join(resoutput_dir,"trainingstim_withinax_svdestdim_df.csv"),index=False)

meanXsvdprojection_df = pd.concat(meanXsvdprojection_dfs,axis=0).reset_index(drop=True)
meanXsvdprojection_df.to_csv(os.path.join(resoutput_dir,"trainingstim_withinax_meanXsvd1Dproj_df.csv"),index=False)



## Run CV-SVD for test stimuli within axis
print("\nRun CV-SVD for test stimuli\n")
testsvdestdim_dfs = []
testsvdprojection_dfs = []
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

                est_d, reconstruction_corr,reconstruction_r2,unique_sess,projection_1D = cross_val_SVD(whitenedX,session_labels=session_labels,flag_centering=False)
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

                # perform SVD on the average matrix
                meanX = np.mean(split_data(whitenedX,session_labels),axis=0)
                u,s,vh = np.linalg.svd(meanX,full_matrices=False)
                # compute the projection of the non-test runs onto the first component and compute the correlation with the stimuli features of the test data
                projection_1D = meanX@vh[0,:]#np.dot(u[:,:1],s[:1]) # projection of the training data onto the first component
                
                tmpprjdfs = [
                    pd.DataFrame(
                    {
                        "SVD_projection": prj,
                        "axisloc": stim_features[session_labels==0],
                    }
                ).assign(run=r) for prj,r in zip([projection_1D],[0])
                ]
                testsvdprojection_dfs.append(pd.concat(tmpprjdfs,axis=0).assign(
                    subid=subid,
                    roi=roi,
                    dimonaxis = axisname,
                    contorlaxisloc = t)
                    )

testsvdestdim_df = pd.concat(testsvdestdim_dfs,axis=0).reset_index(drop=True)
testsvdestdim_df.to_csv(os.path.join(resoutput_dir,"teststim_withinax_svdestdim_df.csv"),index=False)

testsvdprojection_df = pd.concat(testsvdprojection_dfs,axis=0).reset_index(drop=True)
testsvdprojection_df.to_csv(os.path.join(resoutput_dir,"teststim_withinax_svd1Dproj_df.csv"),index=False)
