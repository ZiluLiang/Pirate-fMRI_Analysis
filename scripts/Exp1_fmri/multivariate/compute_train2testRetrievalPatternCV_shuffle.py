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
from scripts.Exp1_fmri.multivariate.pirateOMutils import shuffle_locations

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
    # add a column of stim_attrx and stim_attry for later shuffling
    stimdf["stim_attrx"] = stimdf["stim_x"]*2+2
    stimdf["stim_attry"] = stimdf["stim_y"]*2+2
    stimdf["stim_attrx"] = stimdf["stim_attrx"].astype(int)
    stimdf["stim_attry"] = stimdf["stim_attry"].astype(int)
    assert np.array_equal(stimdf["stim_attrx"].unique(), [0,1,2,3,4])
    assert np.array_equal(stimdf["stim_attry"].unique(), [0,1,2,3,4])

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

def RPCV_run_group(datasetname,roi,run_regs,outputdir,outputprefix=""):
    dataset = load(datasetname)

    print(f"Running ROI: {roi} - regs: {run_regs} - outputprefix: {outputprefix}")
    subXs, subdfs, subid_list = dataset["Xs"], dataset["dfs"], dataset["subids"]
    if outputprefix != "":
        outputprefix = outputprefix + "_"
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
    resultdf.to_csv(os.path.join(outputdir,f"{outputprefix}results.csv"),index=False)
    dump(reg_coefs,os.path.join(outputdir,f"{outputprefix}reg_coefs.pkl"))

    return reg_coefs,resultdf

##################################################### ANALYSIS STARTS HERE #####################################################
organize_and_get_data = False
shuffle_data = False
shuffle_args = {
    "shuffle_function":"factorised", # "fullrand" or "factorised"
    "shared_seeds": True,
    "n_shuffles": 10000
    }
run_ana = False
save_summary = True

rois = ["PPC_bilateral","V1_bilateral","vmPFC_bilateral","HPC_bilateral"]

resultdir = os.path.join(ROIRSAdir,"retrievalpatternCV")
checkdir(resultdir)
resoverview_dir = os.path.join(resultdir,"resoverview")
checkdir(resoverview_dir)    

# get shorthand names for file saving
sf = shuffle_args["shuffle_function"]
ss = shuffle_args["shared_seeds"]
ss_str = 'sharedseeds' if ss else 'diffseeds'
n_shuffles = shuffle_args["n_shuffles"]
if n_shuffles <= 9999:
    ss_str = f"{ss_str}_nshuffles{n_shuffles:04d}"
    shuffleoutputprefix = "shuffle%04d"
elif n_shuffles <= 99999:
    ss_str = f"{ss_str}_nshuffles{n_shuffles:05d}"
    shuffleoutputprefix = "shuffle%05d"
else:
    raise ValueError("n_shuffles is too large, please keep it under 100000")

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

if shuffle_data:
    # we only need to shuffle the locations once and applied it to all the roi
    roi = rois[0]
    tidied_datasets = load(os.path.join(resultdir,roi,"tidied_datasets.pkl"))
    subXs, subdfs, subid_list = tidied_datasets["Xs"], tidied_datasets["dfs"], tidied_datasets["subids"]
    shuffled_dfs_allshuffles = []
    for shuffle_idx in range(n_shuffles):
        shuffled_dfs = []
        for Xs,dfs,subid in zip(subXs,subdfs,subid_list):
            if shuffle_args["shared_seeds"]:
                rs = shuffle_idx
            else:
                rs = int(subid.replace("sub",""))*n_shuffles + shuffle_idx
            print(f"{rs} - {subid}",end="\r",flush=True)
            [training_X,test_X,lzer_X],  [training_df,test_df,lzer_df] = Xs, dfs
            # create copies to avoid modifying the original data
            training_df_shuffled = training_df.copy()
            test_df_shuffled = test_df.copy()
            lzer_df_shuffled = lzer_df.copy()

            # shuffle the test locations first
            stimgtlocs = np.copy(test_df[["stim_x","stim_y"]].to_numpy())
            shufffled_locs, (x_map, y_map) = shuffle_locations(stimgtlocs, shuffle_function=sf, randomseed=rs, return_mapping=True,force_different=True)
            test_df_shuffled[["stim_x","stim_y"]] = shufffled_locs            
            
            x_map[0.0], y_map[0.0] = 0.0, 0.0 # need to keep the central location unchanged
            # then we apply the same x_map and y_map to training stimuli df
            traingtlocs = training_df[["stim_x","stim_y"]].to_numpy()
            shuffled_trlocs = np.array([[x_map[x], y_map[y]] for x, y in traingtlocs])
            training_df_shuffled[["stim_x","stim_y"]] = shuffled_trlocs
            # do the same for localizer stimuli df
            lzertlocs = lzer_df[["stim_x","stim_y"]].to_numpy()
            shuffled_lzerlocs = np.array([[x_map[x], y_map[y]] for x, y in lzertlocs])
            lzer_df_shuffled[["stim_x","stim_y"]] = shuffled_lzerlocs
        
            shuffled_dfs.append([training_df_shuffled,test_df_shuffled,lzer_df_shuffled])
        
        shuffled_dfs_allshuffles.append(shuffled_dfs)

    # apply the shuffled dfs to all rois and save
    for roi in rois:
        shuffled_ds_dir = os.path.join(resultdir,roi,f"shuffled_datasets_{sf}_{ss_str}")
        checkdir(shuffled_ds_dir)

        tidied_datasets = load(os.path.join(resultdir,roi,"tidied_datasets.pkl"))
        subXs, subid_list = tidied_datasets["Xs"], tidied_datasets["subids"]
        for shuffle_idx in range(n_shuffles):
            print(f"Saving shuffled dataset for ROI: {roi} - shuffle: {shuffle_idx}",end="\r",flush=True)
            shuffled_dfs = shuffled_dfs_allshuffles[shuffle_idx]
            shuffled_datasets = {
                "Xs": [],
                "dfs": [],
                "subids": []
                }

            for Xs,dfs,subid in zip(subXs,shuffled_dfs,subid_list):
                shuffled_datasets["Xs"].append(Xs)
                shuffled_datasets["dfs"].append(dfs)
                shuffled_datasets["subids"].append(subid)
            dump(
                shuffled_datasets,
                os.path.join(shuffled_ds_dir,f"{shuffleoutputprefix%shuffle_idx}.pkl")
            )

        
# run the analysis
if run_ana:
    print("\nRunning analysis\n")

    with Parallel(n_jobs=cpu_count()-2,verbose=5) as parallel:
        for roi in rois:
            print(f"Running analysis in ROI: {roi}")

            tidied_ds_name = os.path.join(resultdir,roi,"tidied_datasets.pkl")
            #tidied_datasets = load(tidied_ds_name)
            run_regs = ["train2test"]
            roi_output_dir = os.path.join(resultdir,roi)
            checkdir(roi_output_dir)
            RPCV_run_group(tidied_ds_name,roi,run_regs,roi_output_dir,outputprefix="")
            
            shuffled_ds_dir = os.path.join(roi_output_dir,f"shuffled_datasets_{sf}_{ss_str}")
            shuffled_res_dir = os.path.join(roi_output_dir,f"shuffled_results_{sf}_{ss_str}")
            checkdir(shuffled_res_dir)
            shuffled_ds_names = [os.path.join(shuffled_ds_dir,f"{shuffleoutputprefix%shuffle_idx}.pkl") for shuffle_idx in range(n_shuffles)]
            parallel(delayed(RPCV_run_group)(
                ds,roi,run_regs,shuffled_res_dir,outputprefix=f"{shuffleoutputprefix}"%idx
            ) for idx,ds in enumerate(shuffled_ds_names))

if save_summary:

    # get the null results
    null_res_dfs = []
    for roi in rois:
        print(roi,end="\n")
        shuffled_res_dir = os.path.join(resultdir,roi,f"shuffled_results_{sf}_{ss_str}")
        for shuffle_idx in range(n_shuffles):
            print(f"Processing shuffle {shuffle_idx}",end="\r")
            shufflefn = f"{shuffleoutputprefix}_results.csv"%shuffle_idx
            shuffdf = pd.read_csv(os.path.join(shuffled_res_dir,shufflefn)).assign(roi=roi,shuffle_idx=shuffle_idx)
            shuffdf["subgroup"] = shuffdf.subid.map(lambda x: "Generalizer" if x in generalizers else "nonGeneralizer")
            # get the null for each permutation
            shuffdf_groupsum = shuffdf.groupby(['roi', 'subgroup', 'shuffle_idx','metric'])['value'].mean().reset_index()
            shuffdf_allsum   = shuffdf.groupby(['roi', 'shuffle_idx','metric'])['value'].mean().reset_index().assign(subgroup="all")
            null_res_dfs.append(shuffdf_groupsum)
            null_res_dfs.append(shuffdf_allsum)
    null_res_df = pd.concat(null_res_dfs,axis=0).reset_index(drop=True)
    null_res_df[["analysis","metricname"]] = null_res_df.metric.str.split("-",expand=True)
    
    # get the true the results
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
        "results": resoverview_df,
        "null_results": null_res_df,
        "coefs": reg_coefs
        },
        os.path.join(resoverview_dir,"retrievalpatternCV_results_withnull.pkl")   
    )

plot_summary = True
if plot_summary:
    res = load(os.path.join(resoverview_dir,"retrievalpatternCV_results_withnull.pkl"))
    resoverview_df = res["results"]
    resoverview_df[["analysis","metricname"]] = resoverview_df.metric.str.split("-",expand=True)
    resoverview_df["subgroup"] = ["Generalizer" if subid in generalizers else "nonGeneralizer" for subid in resoverview_df.subid]

    null_res_df = res["null_results"]
    reg_coefs = res["coefs"]
    ##################################################### plot the results for a quick check ##################################################### 
    #resoverview_df["subgroup"] = pd.Categorical(resoverview_df["subgroup"],categories=["Generalizer","nonGeneralizer"],ordered=True)
    checkmetrics = ["eval_corr","eval_r2","compoweight","meanweightdiff","reg_percept","reg_distmod"]
    trtepred_df = resoverview_df[(resoverview_df.analysis=="train2test")&(resoverview_df.metricname.isin(checkmetrics))].copy()
    trtepred_df.metricname = pd.Categorical(trtepred_df.metricname, categories=checkmetrics,ordered=True)
    trtepred_nulldf = null_res_df[(null_res_df.analysis=="train2test")&(null_res_df.metricname.isin(checkmetrics))].copy()
    trtepred_nulldf.metricname = pd.Categorical(trtepred_nulldf.metricname, categories=checkmetrics,ordered=True)

    # plot the distribution of true against null
    # first we create a dataframe that contains both true and null results
    truedf = trtepred_df.assign(resulttype="true",shuffle_idx=-1)
    nulldf = trtepred_nulldf.assign(resulttype="null")
    truedf["roi"] = pd.Categorical(
        truedf.roi.map(lambda x: x.replace("_bilateral","")),
        categories=["V1","PPC","HPC","vmPFC"],
        ordered=True
    )
    nulldf["roi"] = pd.Categorical(
        nulldf.roi.map(lambda x: x.replace("_bilateral","")),
        categories=["V1","PPC","HPC","vmPFC"],
        ordered=True
    )

    xvar, yvar, huevar = "roi", "value", "resulttype"
    colvar, rowvar = "subgroup","metricname"
    palettes = dict(zip(["true","null"],#combined_df[huevar].unique(),
                        ["black","grey"]#sns.color_palette(None,combined_df[huevar].nunique())
                        ))
    ncol, nrow = truedf[colvar].nunique(), truedf[rowvar].nunique()
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow), squeeze=False, sharex=False, sharey=False)
    for j, (gname,gdf_true) in enumerate(truedf.groupby([rowvar,colvar])):
        ax = axes.flatten()[j]
        gdf_null = nulldf[
            (nulldf[colvar]==gname[1]) &
            (nulldf[rowvar]==gname[0])
        ]
        
        sns.violinplot(data=gdf_null, x=xvar, y=yvar, hue=huevar, palette=palettes, 
                       split=True, fill=True, ax=ax,alpha=0.5,
                       bw_adjust=1,
                       inner=None)
        #sns.violinplot(data=gdf_true, x=xvar, y=yvar, hue=huevar, palette=palettes, split=True, fill=False, ax=ax)
        sns.pointplot(data=gdf_true, x=xvar, y=yvar, hue=huevar, palette=palettes, ax=ax, errorbar=None, markers="o", linestyles="")
        if gname[0] in ["reg_percept","reg_distmod"]:
            model_names = {
                "reg_percept":"High-D model (Perceptual)",
                "reg_distmod":"Low-D model (Distance-modulated)",
            }
            ax.set_title(f"{model_names[gname[0]]}\n{gname[1]}", y = 1.1)
        else:
            ax.set_title(f"{gname[0]}\n{gname[1]}", y = 1.05)
        #ax.legend_.remove()
        # add significance stars for reg_percept and reg_distmod only
        if gname[0] in ["reg_percept","reg_distmod"]:
            # do it separately for each roi
            for t in gdf_true.roi.unique():
                gdf_true_roi = gdf_true[gdf_true.roi==t].copy()
                gdf_null_roi = gdf_null[gdf_null.roi==t].copy()    
                truevals = gdf_true_roi.value.values
                nullvals = gdf_null_roi.value.values
                pval = 1-np.mean(np.mean(truevals)>nullvals)
                # add stars
                if pval < 0.001:
                    star = '***'
                elif pval < 0.01:
                    star = '**'
                elif pval < 0.05:
                    star = '*'
                else:
                    star = ''
                i = list(gdf_true.roi.unique()).index(t)
                ax.text(i, np.max(nullvals) + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), star, ha='center', va='bottom', color='k', fontsize=14)
        sns.move_legend(ax, "lower center", title="", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    