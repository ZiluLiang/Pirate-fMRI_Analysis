"""
This file computes the retrieval pattern of test stimuli from training stimuli/location.
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

from sklearn.linear_model import LinearRegression
import scipy.spatial
import scipy.stats

import sys
project_path = r"D:\OneDrive - Nexus365\pirate_ongoing"
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim
from scripts.Exp1_fmri.multivariate.mvpa_estimator import CompositionalRetrieval
from sklearn.linear_model import LinearRegression
import scipy.spatial
import scipy.stats

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

ROIRSAdir = os.path.join(project_path,'AALandHCPMMP1andFUNCcluster')
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois = list(roi_data.keys())

reg_coefs,percept_reg_coefs = dict(zip(["train2test","loc2test","loc2train"],[{},{},{}])), dict(zip(["train2test","loc2test","loc2train"],[{},{},{}]))
dmod_reg_coefs = dict(zip(["train2test","loc2test","loc2train"],[{},{},{}]))
compositionretrieval_results = []
for roi in rois:
    # initialize the dictionary to store the results
    for ana in ["train2test","loc2test","loc2train"]:
        reg_coefs[ana][roi],      percept_reg_coefs[ana][roi]  = {"Generalizer":[],"nonGeneralizer":[]}, {"Generalizer":[],"nonGeneralizer":[]}
        dmod_reg_coefs[ana][roi] = {"Generalizer":[],"nonGeneralizer":[]}
    
    # run the analysis for each subject
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)
        subgroup = "Generalizer" if subid in generalizers else "nonGeneralizer"
        preprocedX = deepcopy(subdata["preprocX"])
        stimdf = subdata["stimdf"]
        
        navi_filter = np.vstack(
            [stimdf.taskname.to_numpy() == "navigation",
             [not np.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]]
        ).all(axis=0)
        
        #average pirate task across sessions
        navi_X = np.mean(split_data(X      = preprocedX[navi_filter,:],
                                    groups = stimdf[navi_filter].copy().stim_session.to_numpy()),
                        axis=0)
        
        assert navi_X.shape[0] == 24
        navi_df = stimdf[navi_filter&(stimdf.stim_session==0)].copy().reset_index(drop=True)

        training_filter = navi_df.stim_group==1
        test_filter = navi_df.stim_group==0

        trdf = navi_df[training_filter].copy().reset_index(drop=True)
        trX  = navi_X[training_filter]
        trdfneworder = trdf.sort_values(by=['training_axset','training_axlocTL'])
        new_order = trdfneworder.index
        training_df = trdfneworder.reset_index(drop=True).assign(stim_task=0)
        training_X = trX[new_order,:]
        
        
        test_X = navi_X[test_filter]
        test_df = navi_df[test_filter].copy().reset_index(drop=True).assign(stim_task=0)


        lzer_filter = np.vstack(
            [stimdf.taskname.to_numpy() == "localizer",
             [not np.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]]
        ).all(axis=0)
        lzerX  = preprocedX[lzer_filter,:]
        lzdfneworder = stimdf[lzer_filter].copy().reset_index(drop=True).sort_values(by=['training_axset','training_axlocTL'])
        new_order = lzdfneworder.index
        lzer_df = lzdfneworder.reset_index(drop=True).assign(stim_task=1)
        lzer_X = lzerX[new_order,:]

        CRestimator = CompositionalRetrieval(
                               activitypattern=np.vstack([training_X,test_X,lzer_X]),
                               stim_df=pd.concat([training_df,test_df,lzer_df],axis=0).reset_index(drop=True)
                               )
        CRestimator.fit()

        subresdf = pd.DataFrame({"value":CRestimator.result,
                                 "metric":CRestimator.resultnames}
                                 ).assign(roi=roi,subid=subid,subgroup=subgroup)
        compositionretrieval_results.append(subresdf)
        for ana in ["train2test","loc2test","loc2train"]:
            reg_coefs[ana][roi][subgroup].append(CRestimator.regcoefmats[ana][0])
            percept_reg_coefs[ana][roi][subgroup].append(CRestimator.regcoefmats[ana][1]) 
            dmod_reg_coefs[ana][roi][subgroup].append(CRestimator.regcoefmats[ana][2])
        
compositionretrieval_resdf = pd.concat(compositionretrieval_results,axis=0).reset_index(drop=True)

dump({"results":compositionretrieval_resdf,
      "fit_results":reg_coefs,           # The fitted retrieval weights matrices (regression coefficients from LinearRegression()) for each participant
      "percept_model":percept_reg_coefs, # The perceptual matching model weight matrix: this should be the same for each participant, still save it to double check
      "dmod_model":dmod_reg_coefs        # The distance modulation model weight matrix: this should be the same for each participant, still save it to double check
      },
      os.path.join(ROIRSAdir,"ROI_RetrievalPattern.pkl")
      )