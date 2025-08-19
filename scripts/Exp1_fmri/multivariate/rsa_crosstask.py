

import numpy as np
import scipy
import scipy.spatial
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import json
from copy import deepcopy
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri, compute_rdm, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature,concat_data,kabsch_algorithm
from zpyhelper.MVPA.estimators import MultipleRDMRegression, PatternCorrelation

from typing import Union
import sys
import os


project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')
sys.path.append(project_path)

from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM

import warnings
warnings.simplefilter('ignore', category=FutureWarning)


# Load data
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois = ["V1_bilateral","PPC_bilateral","vmPFC_bilateral","HPC_bilateral"]


############################################### CROSS-TASK RSA ANALYSIS ########################################################
trstim_rdms,testim_rdms, lzerstim_rdms = {}, {}, {}
crosstask_rdms = {"symmetric":{}, "asymmetric":{}}
filtered_roi_data = {roi: [] for roi in rois}
crosstaskrsa_df_list = []
for roi in rois:
    trstim_rdms[roi], testim_rdms[roi], lzerstim_rdms[roi] = {"stim_x":[],"stim_y":[]}, {"stim_x":[],"stim_y":[]}, {"stim_x":[],"stim_y":[]}
    crosstask_rdms["symmetric"][roi] = []
    crosstask_rdms["asymmetric"][roi] = []
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)
        subgroup = "Generalizer" if subid in generalizers else "nonGeneralizer"
        preprocedX = deepcopy(subdata["preprocX"])#deepcopy(subdata["rawX"])#
        stimdf = subdata["stimdf"].copy()
        

        #get treasure hunt data and average across sessions
        navi_filter = stimdf.taskname.to_numpy() == "navigation"
        non_center_filter = [not all([x==0, y==0]) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]
        ncnavi_filter = np.all([navi_filter,non_center_filter],axis=0)
        navi_X = np.mean(split_data(X      = preprocedX[ncnavi_filter,:],
                                    groups = stimdf[ncnavi_filter].copy().stim_session.to_numpy()),
                        axis=0)
        navi_stim_df = stimdf[ncnavi_filter&(stimdf.stim_session==0)].copy().reset_index(drop=True)
        
        # get localizer task data
        lzer_filter = stimdf.taskname.to_numpy() == "localizer"
        non_center_filter = [not all([x==0, y==0]) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]
        nclzer_filter = np.all([lzer_filter,non_center_filter],axis=0)
        lzer_X = preprocedX[nclzer_filter,:]
        lzer_stim_df = stimdf[nclzer_filter].copy().reset_index(drop=True)
        
        # Run cross-task RSA
        ### compute a cross task RDM for train2lzer
        train_X = navi_X[navi_stim_df.stim_group == 1,:]
        traindf = navi_stim_df[navi_stim_df.stim_group == 1].copy().reset_index(drop=True)
        reorder_train = traindf.sort_values(["training_axset","training_axlocTL"]).index.to_numpy()
        reorder_lzer = lzer_stim_df.sort_values(["training_axset","training_axlocTL"]).index.to_numpy()
        traindf_reordered = traindf.iloc[reorder_train,:].copy().reset_index(drop=True)
        lzerdf_reordered = lzer_stim_df.iloc[reorder_lzer,:].copy().reset_index(drop=True)
        train2lzer_crosstaskrdm_asym = scipy.spatial.distance.cdist(
            train_X[reorder_train,:],
            lzer_X[reorder_lzer,:],
            metric = "correlation"
        )        
        train2lzer_crosstaskrdm_sym = (train2lzer_crosstaskrdm_asym + train2lzer_crosstaskrdm_asym.T)/2 # make it symmetric
        crosstask_rdms["asymmetric"][roi].append(train2lzer_crosstaskrdm_asym)
        crosstask_rdms["symmetric"][roi].append(train2lzer_crosstaskrdm_sym)
        ### see if:
        ### 1) same-to-same pairs are less different than different pairs
        same2same = np.diag(train2lzer_crosstaskrdm_sym).mean()
        same2diff = lower_tri(train2lzer_crosstaskrdm_sym)[0].mean()
        ssrdmcorr = scipy.stats.spearmanr(
                            train2lzer_crosstaskrdm_sym.flatten(),
                            1-np.eye(train2lzer_crosstaskrdm_sym.shape[0]).flatten() # 1 minus a identity matrix, yielding a matrix only off-diagonal is 1
                              ).statistic
        ### 2) regression with feature based + distance based models
        feat_model = scipy.spatial.distance.cdist(
            traindf_reordered[["stim_x","stim_y"]].to_numpy(),
            traindf_reordered[["stim_x","stim_y"]].to_numpy(),
            metric = "hamming"
        )
        euc_model = scipy.spatial.distance.cdist(
            traindf_reordered[["stim_x","stim_y"]].to_numpy(),
            traindf_reordered[["stim_x","stim_y"]].to_numpy(),
            metric = "euclidean"
        )
        Y = scale_feature(lower_tri(train2lzer_crosstaskrdm_sym)[0]) # to compute the lower triangle of the rdm
        X = np.vstack([scale_feature(lower_tri(feat_model)[0]), 
                       scale_feature(lower_tri(euc_model)[0]) ]).T
        sym_feateuc_coef = LinearRegression().fit(
            y=Y,
            X=X
        ).coef_
        
        crosstaskrsa_df_list.append(
            pd.DataFrame(
                        {"modelrdm":   ["patcorr_same2same","patcorr_same2diff", "patcorr_SSminSD",
                                        "same2same",
                                        "high D","low D"],
                        "coefficient": [1-same2same,  1-same2diff, -same2same+same2diff,
                                        ssrdmcorr,
                                        sym_feateuc_coef[0],sym_feateuc_coef[1]
                                        ],
                        "analysis": ["meancrosstaskpatcorr","meancrosstaskpatcorr","meancrosstaskpatcorrdiff",
                                     "corr_same2samerdm2asymcrosstaskrdm",
                                     "compare high low D (sym)","compare high low D (sym)"]}
                    ).assign(
                        roi = roi,
                        subid=subid,
                        subgroup=subgroup
                    )
        )

                
crosstaskrsa_df = pd.concat(crosstaskrsa_df_list, axis=0).reset_index(drop=True)
checkdir(os.path.join(ROIRSAdir,"crosstaskrsa"))
crosstaskrsa_df.to_csv(os.path.join(ROIRSAdir,"crosstaskrsa","crosstaskrsa_df.csv"),index=False)
dump(crosstask_rdms,filename=os.path.join(ROIRSAdir,"crosstaskrsa","crosstask_rdms.pkl"))


#################### plot the results###########################
# for analysis 3
gs = sns.catplot(
    data = crosstaskrsa_df,
    x = "modelrdm",
    y = "coefficient",
    hue = "subgroup",
    palette={"Generalizer":"#1f77b4","nonGeneralizer":"#ff7f0e"},
    col="roi",
    row="analysis",
    kind="point",errorbar="se",capsize=0.1,dodge=0.5,
    sharex=False, sharey=False
).refline(y=0, linestyle="--", color="black").set_titles("{row_name} \n {col_name}",fontweight="bold",fontsize=12)        
gs.map_dataframe(
    sns.swarmplot,
    x="modelrdm", 
    y="coefficient",
    hue="subgroup",
    palette={"Generalizer":"#1f77b4","nonGeneralizer":"#ff7f0e"},
    dodge=0.4,
    alpha=0.5,

)


### plot the group average RDM
fig,axes = plt.subplots(2,4,figsize=(12,6))
for j,roi in enumerate(rois):
    G_meanrdm = [x for x,subid in zip(crosstask_rdms["symmetric"][roi],subid_list) if subid in generalizers]
    NG_meanrdm = [x for x,subid in zip(crosstask_rdms["symmetric"][roi],subid_list) if subid in nongeneralizers]
    sns.heatmap(
        np.mean(G_meanrdm,axis=0),
        ax=axes[0,j],
        annot=False
    )
    sns.heatmap(
        np.mean(NG_meanrdm,axis=0),
        ax=axes[1,j],
        annot=False
    )
    axes[0,j].set_title(f"{roi.replace('_bilateral','')} \n Generalizers",fontweight="bold")
    axes[1,j].set_title(f"{roi.replace('_bilateral','')} \n nonGeneralizers",fontweight="bold")
fig.tight_layout()

### Sanity Check: only plot the good participants in PPC
fig,axes = plt.subplots(2,4,figsize=(12,6))
curr_roi_df = crosstaskrsa_df[(crosstaskrsa_df.analysis=="compare high low D (sym)")&(crosstaskrsa_df.modelrdm=="low D")&(crosstaskrsa_df.roi==roi)].copy()
curr_roi_G_df = curr_roi_df[curr_roi_df.subgroup=="Generalizer"].copy()
curr_roi_NG_df = curr_roi_df[curr_roi_df.subgroup=="nonGeneralizer"].copy()

max_corr_G = curr_roi_G_df.coefficient.max()
max_corr_NG = curr_roi_NG_df.coefficient.max()

for j,roi in enumerate(rois):
    max_corr_G_subid = curr_roi_G_df[curr_roi_G_df.coefficient>0].subid.values
    max_corr_NG_subid = curr_roi_NG_df[curr_roi_NG_df.coefficient>0].subid.values

    sns.heatmap(
        np.mean([x for x,subid in zip(crosstask_rdms["symmetric"][roi],subid_list) if subid in max_corr_G_subid],axis=0),
        ax=axes[0,j],
        annot=False
    )
    sns.heatmap(
        np.mean([x for x,subid in zip(crosstask_rdms["symmetric"][roi],subid_list) if subid in max_corr_NG_subid],axis=0),        
        ax=axes[1,j],
        annot=False
    )

    axes[0,j].set_title(f"{roi.replace('_bilateral','')} \n Generalizer")
    axes[1,j].set_title(f"{roi.replace('_bilateral','')} \n nonGeneralizer")
fig.tight_layout()
    