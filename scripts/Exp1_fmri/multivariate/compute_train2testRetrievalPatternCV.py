"""
This file computes the retrieval pattern of test stimuli from training stimuli/location with cross-validation.
"""
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import json
from copy import deepcopy
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri, compute_rdm, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature,concat_data

from typing import Union
import sys
import os


project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')

sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.mvpa_estimator import CompositionalRetrieval_CV


import warnings
warnings.simplefilter('ignore', category=FutureWarning)


# Load data
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

##################################################### METHODS #####################################################

def average_and_reorder(activitypattern:np.array,stimdf:pd.DataFrame)-> Union[list, list]:
    """Take the average activity pattern of odd and even runs (excluding the central training stimulus),\
    Split into training and test stimuli, reorder according to features

    Parameters
    ----------
    activitypattern : np.array
        activity patterns, shape (nstimuli, nvoxel)
    stimdf : pd.DataFrame
        dataframe containing stimuli information

    Returns
    -------
    Union[list, list]
        return a list of activity pattern and a list of stimuli dataframe in the order of [training, test, localizer]
    """
        
    # filter out data from treasure hunt task
    navi_filter = np.vstack(
            [stimdf.taskname.to_numpy() == "navigation",
            [not np.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]] # exclude central stimuli
    ).all(axis=0)

    navi_X = activitypattern[navi_filter,:]
    assert navi_X.shape[0] == 24*4 # because we are ignoring the central sitmuli here, so we have (25-1)*4 stimuli in total
    navi_df = stimdf[navi_filter].copy().reset_index(drop=True)

    #then we process the treasure hunt task data here: we get the average of odd and even splits
    navi_Xsess = split_data(navi_X,navi_df.stim_session.to_numpy())
    oddX,evenX = np.mean([navi_Xsess[0],navi_Xsess[2]],axis=0), np.mean([navi_Xsess[1],navi_Xsess[3]],axis=0)
    # then we z-score for each split
    oddX,evenX = scale_feature(oddX,2), scale_feature(evenX,2)
    # and we put them back into a matrix
    navi_X = concat_data([oddX, evenX])
    navi_df = navi_df[navi_df.stim_session<2].copy().reset_index(drop=True)

    ## split into training and test for reordering: this is to make sure the final weight matrices are ordered in the way we want
    training_filter = navi_df.stim_group==1
    test_filter = navi_df.stim_group==0


    trdf = navi_df[training_filter].copy().reset_index(drop=True)
    trX  = navi_X[training_filter]
    trdfneworder = trdf.sort_values(by=['stim_session','training_axset','training_axlocTL'])
    trneworderidx = trdfneworder.index
    training_X = trX[trneworderidx,:]
    training_df = trdfneworder.reset_index(drop=True).assign(stim_task=0)
    
    teX = navi_X[test_filter]
    tedf = navi_df[test_filter].copy().reset_index(drop=True)
    teneworder = tedf.sort_values(by=['stim_session','stim_x','stim_y'])
    teneworderidx = teneworder.index
    test_X = teX[teneworderidx,:]
    test_df = teneworder.reset_index(drop=True).assign(stim_task=0)

    # filter out data from localizer task
    lzer_filter = np.vstack(
        [stimdf.taskname.to_numpy() == "localizer",
        [not np.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]]
    ).all(axis=0)       
    lzerX  = activitypattern[lzer_filter,:]
    # then we z-score for each split (only 1 split for localizer)
    lzerX = scale_feature(lzerX,2)
    lzdfneworder = stimdf[lzer_filter].copy().reset_index(drop=True).sort_values(by=['stim_session','training_axset','training_axlocTL'])
    new_order = lzdfneworder.index
    lzer_df = lzdfneworder.reset_index(drop=True).assign(stim_task=1)
    lzer_X = lzerX[new_order,:]
    
    return [training_X,test_X,lzer_X],  [training_df,test_df,lzer_df]

##################################################### ANALYSIS STARTS HERE #####################################################
organize_and_get_data = True
run_ana = True
rois = ["PPC_bilateral","V1_bilateral","vmPFC_bilateral","HPC_bilateral"]
resultdir = os.path.join(ROIRSAdir,"retrievalpatternCV")
checkdir(resultdir)


if organize_and_get_data:
    roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
    #rois = list(roi_data.keys())
    
    # first we get the averaged and reordered data
    allsubXs, allsubdfs = {},{}
    for roi in rois:
        allsubXs[roi], allsubdfs[roi] = [],[]
        for subdata,subid in zip(roi_data[roi],subid_list):
            print(f"{roi} - {subid}",end="\r",flush=True)
            # reorder and get the data for each pariticpnat
            Xs, dfs = average_and_reorder(deepcopy(subdata["preprocX"]),subdata["stimdf"].copy())
            allsubXs[roi].append(Xs)
            allsubdfs[roi].append(dfs)
            
        checkdir(os.path.join(resultdir,roi))
        print(f"\nSaving tidied data for {roi}")
        dump({"Xs":allsubXs[roi],
              "dfs":allsubdfs[roi],
              "subids":subid_list
            },
            os.path.join(resultdir,roi,f"tidied_datasets.pkl")
            )



# run the analysis
if run_ana:
    print("\nRunning analysis\n")
    def RPCV_run_group(resultdir,roi,run_regs):
        tidied_datasets = load(os.path.join(resultdir,roi,"tidied_datasets.pkl"))
        subXs, subdfs, subid_list = tidied_datasets["Xs"], tidied_datasets["dfs"], tidied_datasets["subids"]
        
        results = []
        reg_coefs = dict(zip(run_regs,[[] for _ in run_regs]))
        
        for Xs,dfs,subid in zip(subXs,subdfs,subid_list):
            CRestimator = CompositionalRetrieval_CV(
                                activitypatterns=Xs,
                                stim_dfs=dfs,
                                run_reg=run_regs
                                )
            CRestimator.fit()

            subresdf = pd.DataFrame({"value":CRestimator.result,
                                    "metric":CRestimator.resultnames}
                                    ).assign(roi=roi,subid=subid)
            results.append(subresdf)
            for ana in run_regs:
                reg_coefs[ana].append(CRestimator.regcoefmats[ana][0])
        
        # save the model retrieval pattern from last participant for double-checking
        reg_coefs["models"] = {} 
        for ana in run_regs:
            reg_coefs["models"][ana] = [CRestimator.regcoefmats[ana][1], CRestimator.regcoefmats[ana][2]]# in the order of [percept , dismod]
            
        resultdf = pd.concat(results,axis=0).reset_index(drop=True)

        resultdf = resultdf.assign(roi=roi)
        resultdf.to_csv(os.path.join(resultdir,roi,"results.csv"),index=False)
        dump(reg_coefs,os.path.join(resultdir,roi,"reg_coefs.pkl"))

        return reg_coefs,resultdf


    for roi in rois:
        run_regs = ["train2test"]
        RPCV_run_group(resultdir,roi,run_regs)


            
# save the results
resoverview_dir = os.path.join(resultdir,"resoverview")
checkdir(resoverview_dir)    

resoverview_dfs = []
reg_coefs = dict(zip(rois, [{} for _ in rois]))
for roi in rois:
    print(roi,end="\n")
    resultsdf = pd.read_csv(os.path.join(resultdir,roi,"results.csv")).assign(roi=roi)
    resoverview_dfs.append(
        resultsdf
    )

    reg_coefs[roi] = load(os.path.join(resultdir,roi,"reg_coefs.pkl"))

resoverview_df = pd.concat(resoverview_dfs,axis=0).reset_index(drop=True)
resoverview_df[["analysis","metricname"]] = resoverview_df.metric.str.split("-",expand=True)
resoverview_df["subgroup"] = ["Generalizer" if subid in generalizers else "nonGeneralizer" for subid in resoverview_df.subid]

dump(
    {
       "results": pd.concat(resoverview_dfs).reset_index(drop=True),
       "coefs": reg_coefs
    },
    os.path.join(resoverview_dir,"retrievalpatternCV_results.pkl")   
)

##################################################### plot the results for a quick check ##################################################### 
resoverview_df["subgroup"] = pd.Categorical(resoverview_df["subgroup"],categories=["Generalizer","nonGeneralizer"],ordered=True)

checkmetrics = ["eval_corr","eval_r2","compoweight","meanweightdiff","reg_percept","reg_distmod"]


trtepred_df = resoverview_df[(resoverview_df.analysis=="train2test")&(resoverview_df.metricname.isin(checkmetrics))].copy()
trtepred_df.metricname = pd.Categorical(trtepred_df.metricname, categories=checkmetrics,ordered=True)
xvar, yvar, huevar = "subgroup", "value", "roi"
palettes = dict(zip(trtepred_df[huevar].unique(),
                    sns.color_palette(None,trtepred_df[huevar].nunique())
                    ))
gs = sns.catplot(data = trtepred_df,
                x=xvar,y=yvar,hue=huevar,palette = palettes, sharex=False,sharey=False,
                col="metricname",col_wrap=2, 
                kind="violin",split=True,fill=False).set_titles("{col_name}")
gs.map_dataframe(sns.stripplot,x=xvar,y=yvar,hue=huevar,palette = palettes,dodge=True)
gs.fig.suptitle(f"Train to Test Retrieval Performance",y=1.05)
for gname,gdf in trtepred_df.groupby(["metricname","roi","subgroup"]):
    print(gname)
    print(scipy.stats.ttest_1samp(gdf.value.values,0))
gs.savefig(os.path.join(resultdir,"resoverview",f"train2test.png"))


