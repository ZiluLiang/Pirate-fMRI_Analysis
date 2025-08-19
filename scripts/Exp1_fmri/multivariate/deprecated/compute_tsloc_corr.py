import numpy as np
import scipy
import json
import time
import pandas as pd
import glob
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump,load

import scipy.stats
from zpyhelper.MVPA.preprocessors import scale_feature, split_data, concat_data,extract_pc
from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import compute_rdm, lower_tri,compute_rdm_nomial,compute_rdm_residual

from scipy.spatial.distance import cdist,squareform

study_scripts   = r"D:\OneDrive - Nexus365\pirate_ongoing\scripts\Exp1_fmri"

with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]

print("N_Total: ",len(subid_list))

ROIRSAdir = r"D:\OneDrive - Nexus365\pirate_ongoing\AALandHCPMMP1andFUNCcluster"
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois =  list(roi_data.keys())
rois =  [x for x in list(roi_data.keys()) if "bilateral" in x] + ["allgtlocPrecentral_left"]

resoutput_dir = os.path.join(ROIRSAdir,"tsloc_rdmcorr")
checkdir(resoutput_dir)
save_fn = "TSLZERcorr"

def compare_tsloc_rdm(tsrdm,locrdm,eucrdm):
    sim =  scipy.stats.spearmanr(
        lower_tri(tsrdm)[0], lower_tri(locrdm)[0]
    ).statistic
    tssim2euc = scipy.stats.spearmanr(
        lower_tri(tsrdm)[0], lower_tri(eucrdm)[0]
        ).statistic
    locsim2euc = scipy.stats.spearmanr(
        lower_tri(locrdm)[0], lower_tri(eucrdm)[0] 
    ).statistic
    return sim, tssim2euc, locsim2euc

def compute_cross_tsloc(tsX,locX,tsY,locY):
    crosscorr = 1-cdist(tsX,locX,metric="correlation")
    groundtruth = np.argmax(np.array([[1*(y1==y2) for y2 in locY] for y1 in tsY]),axis=1)   	
    clf_res = np.argmax(crosscorr,axis=1)    
    acc = 1*(groundtruth == clf_res)
    return crosscorr, np.mean(acc)

from sklearn.linear_model import LinearRegression
def compute_compordm(locX,locdf,tsX,tsdf,return_rdms = False):
    x_labs = locdf.stim_x.to_numpy()
    y_labs = locdf.stim_y.to_numpy()

    compo_labs = tsdf[["stim_x","stim_y"]].to_numpy()

    compo_features_sep = [
        [np.mean(locX[x_labs==x],axis=0) for x in compo_labs[:,0]],
        [np.mean(locX[y_labs==y],axis=0) for y in compo_labs[:,1]]
    ]
    # the x rdm
    x_rdm = compute_rdm(compo_features_sep[0],metric="correlation")
    y_rdm = compute_rdm(compo_features_sep[1],metric="correlation")
    pred_locrdm = x_rdm+y_rdm

    tsrdm = compute_rdm(tsX,metric="correlation")
    ctrl_rdm = compute_rdm_nomial(compo_labs)

    tar = scale_feature(lower_tri(tsrdm)[0],2)
    tar_resid = compute_rdm_residual(tsrdm,ctrl_rdm,squareform=False)
    
    #x_rdm_resid = compute_rdm_residual(x_rdm,ctrl_rdm,squareform=False)
    #y_rdm_resid = compute_rdm_residual(x_rdm,ctrl_rdm,squareform=False)
    #X_resid = np.vstack([x_rdm_resid,y_rdm_resid]).T


    # run regression on separate rdms
    X = np.vstack([lower_tri(x_rdm)[0],lower_tri(y_rdm)[0]]).T
    X = scale_feature(X,1)
    xycoefs = LinearRegression().fit(X,scale_feature(tar_resid,2)).coef_
    
    # run regression on composed rdms
    #X = np.vstack([lower_tri(pred_locrdm)[0],lower_tri(ctrl_rdm)[0]]).T
    X = scale_feature(X,1)
    compocoefs = LinearRegression().fit(X,tar).coef_
    
    pred_locrdm_resid = compute_rdm_residual(pred_locrdm,ctrl_rdm,squareform=False)
    compocoefs = [scipy.stats.spearmanr(tar_resid,pred_locrdm_resid).statistic]
    

    #pred_locrdm_resid = compute_rdm_residual(pred_locrdm,ctrl_rdm,squareform=False)
    #X_resid = np.atleast_2d(pred_locrdm_resid).T   
    
    if return_rdms:
        return x_rdm, y_rdm, tsrdm, squareform(tar_resid), np.concatenate([xycoefs,compocoefs])
    else:
        return np.concatenate([xycoefs,compocoefs])

TR_corr_res = []
TE_corr_res = []
cross_corr_res = []
TEcompocorr_res = []
for roi in rois:
    tmpTEcompocorr = []
    TRtmpres = [] 
    TEtmpres = {"stim_x":[],"stim_y":[]}
    tmp_cross_corr = {"stim_x":[],"stim_y":[],"training":[]}
    for subdata, subid in zip(roi_data[roi],subid_list):
        non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
        center_filter = [all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
        ncXs, ncDF = subdata["preprocX"][non_center_filter], subdata["stimdf"][non_center_filter].copy().reset_index(drop=True)
        cXs, cDF = subdata["preprocX"][center_filter], subdata["stimdf"][center_filter].copy().reset_index(drop=True)
        session_labels = ncDF.stim_session.to_numpy()
        
        # get filters            
        ## for navigation task
        navi_filter = ncDF.taskname.to_numpy() == "navigation"
        training_filter = np.vstack(
            [ncDF.stim_group.to_numpy() == 1,
             navi_filter]
        ).all(axis=0)
        test_filter = np.vstack(
            [ncDF.stim_group.to_numpy() == 0,
             navi_filter]
        ).all(axis=0)
        # for localizer task
        lzer_filter = ncDF.taskname.to_numpy() == "localizer"
        
        # first we compare the rdm of training stimuli from navigation task and localizer stimuli (pirate appearing at training locations)
        # we reordered the data to make sure the entries in these rdms represent consistent pairs
        lzerX = ncXs[lzer_filter]
        lzerDF = ncDF[lzer_filter].copy().reset_index(drop=True)
        reordered_lzerDF = lzerDF.sort_values(by=["training_axset","training_axlocTL"],inplace=False)
        idx_lzer = reordered_lzerDF.index

        trX = np.mean(split_data(ncXs[training_filter],session_labels[training_filter]),axis=0)
        trDF = ncDF[(training_filter)&(ncDF.stim_session==0)].copy().reset_index(drop=True)
        reordered_trDF = trDF.sort_values(by=["training_axset","training_axlocTL"],inplace=False)
        idx_tr = reordered_trDF.index

        lzer_rdm = compute_rdm(lzerX[idx_lzer],metric="correlation")
        tr_rdm   = compute_rdm(trX[idx_tr],metric="correlation")

        eucrdm = compute_rdm(reordered_trDF[["stim_x","stim_y"]].to_numpy(),metric="euclidean")
        
        TRtmpres.append(compare_tsloc_rdm(tr_rdm,lzer_rdm,eucrdm))

        tmp_cross_corr["training"].append([compute_cross_tsloc(
            trX[idx_tr],
            lzerX[idx_lzer],
            reordered_trDF["stim_id"].values,
            reordered_lzerDF["stim_id"].values
            )[1]])

        #compo rdm for test
        te_fil = np.vstack([navi_filter,test_filter]).all(axis=0)        
        teXave = np.mean(split_data(ncXs[te_fil],ncDF[te_fil].copy().stim_session.to_numpy()),axis=0)
        xycoefs = compute_compordm(lzerX,lzerDF,
                         teXave,ncDF[(te_fil)&(ncDF.stim_session==0)].copy())
        tmpTEcompocorr.append(xycoefs)

        #for test stimuli, we do it for x and y separately, so we need filters for x-axis only and y-axis only stimuli in localizer task
        te_fil = np.vstack([navi_filter,test_filter]).all(axis=0)        
        filter_x = ncDF.stim_x.to_numpy() != 0
        filter_y = ncDF.stim_y.to_numpy() != 0  
        
        for tarvar in ["stim_x","stim_y"]:
            # get splitter
            if tarvar=="stim_x":
                splitvar = "stim_y"
                lzer_fil = np.vstack([lzer_filter,filter_x]).all(axis=0)
            elif tarvar=="stim_y":
                splitvar = "stim_x"
                lzer_fil = np.vstack([lzer_filter,filter_y]).all(axis=0)
                
            # we then further split the data according to the splitvar:
            # if tarvar is "stim_x", we split the data according to the y-axis value
            # in te_Xs/te_DFs, each element is the activity patterns of all x-location for a given y location
            # e.g. te_Xs[0] is the activity patterns of [-2,1], [-1,1], [1,1], [2,1] across all four runs
            te_Xs, splitgroups  = split_data(ncXs[te_fil], ncDF[te_fil].copy()[splitvar].to_numpy(),return_groups=True)
            te_DFs = [ ncDF[te_fil].copy()[ncDF[te_fil].copy()[splitvar].values==g] for g in splitgroups]
            # for localizer task, 
            # we only take the activity patterns of x-axis stimuli if current target is stim_x [-2,0], [-1,0], [1,0] ,[2,0]
            # and y-axis stimuli if current target is stim_y
            lz_X,lz_DF = ncXs[lzer_fil], ncDF[lzer_fil].copy().reset_index(drop=True)

            tmpteres = []
            for te_X,te_DF in zip(te_Xs,te_DFs):
                Xs, locations = split_data(te_X,te_DF[tarvar].to_numpy(),return_groups=True)
                te_X = np.mean(Xs,axis=1)
                lz_X = np.vstack([np.mean(lz_X[lz_DF[tarvar]==l],axis=0) for l in locations])
                res = compare_tsloc_rdm(compute_rdm(te_X,metric="correlation"),
                                        compute_rdm(lz_X,metric="correlation"),
                                        compute_rdm(np.atleast_2d(locations).T,metric="euclidean"))
                tmpteres.append(res)

            TEtmpres[tarvar].append(np.mean(tmpteres,axis=0))# tmpteress is a list of 4 element, each is [tslocsim,tssim2euc,locsim2euc]

            # Xs, locations = split_data(ncXs[te_fil], ncDF[te_fil].copy()[tarvar].values,return_groups=True)
            # te_X = np.mean(Xs,axis=1)
            # lz_X = np.vstack([np.mean(lz_X[lz_DF[tarvar]==l],axis=0) for l in locations])
            # res = compare_tsloc_rdm(compute_rdm(te_X,metric="correlation"),
            #                         compute_rdm(lz_X,metric="correlation"),
            #                         compute_rdm(np.atleast_2d(locations).T,metric="euclidean"))
            # TEtmpres[tarvar].append(res)
            
            tmp_cross_corr[tarvar].append([ compute_cross_tsloc(ncXs[te_fil],
                                                                lz_X,
                                                                ncDF[te_fil].copy()[tarvar].values,
                                                                lz_DF[tarvar].values
                                                                )[1] ])
    TEcompocorr_res.append(#["xcoef","ycoef","ftcoef1","compocoef","ftcoef2"]
        pd.DataFrame(tmpTEcompocorr,columns=["xcoef","ycoef","compocoef"]).assign(subid=subid_list,roi=roi)
    )

    crossres = pd.concat(
        [pd.DataFrame(v,columns=["corrclfacc"]).assign(subid=subid_list,tarvar=k) for k,v in tmp_cross_corr.items()]        
        ).reset_index(drop=True)
    cross_corr_res.append(crossres.assign(roi=roi))

    TRresdf = pd.DataFrame(TRtmpres,columns=["tssim2loc","tssim2euc","locsim2euc"])
    TRresdf["subid"] = subid_list
    TR_corr_res.append(
        TRresdf.assign(roi=roi)
        )
    
    TEresdf = pd.concat(
        [pd.DataFrame(v,columns=["tssim2loc","tssim2euc","locsim2euc"]).assign(subid=subid_list,tarvar=k) for k,v in TEtmpres.items()]
    ).reset_index(drop=True)
    TE_corr_res.append(TEresdf.assign(roi=roi))
    
TEcompocorr_res_df = pd.concat(TEcompocorr_res).reset_index(drop=True)
TRcorr_res_df = pd.concat(TR_corr_res).reset_index(drop=True)
TEcorr_res_df = pd.concat(TE_corr_res).reset_index(drop=True)
cross_corr_res_df = pd.concat(cross_corr_res).reset_index(drop=True)

dump(
    {"TEcompoLZERcorr":TEcompocorr_res_df,
     "TRLZERcorr":TRcorr_res_df,
     "TELZERcorr":TEcorr_res_df,
     "corrclfacc":cross_corr_res_df
     },
     os.path.join(resoutput_dir,f"{save_fn}.pkl")
)


# plot the results
loadedres = load(os.path.join(resoutput_dir,f"{save_fn}.pkl"))
TRcorr_res_df = loadedres["TRLZERcorr"]
TEcorr_res_df = loadedres["TELZERcorr"]
cross_corr_res_df = loadedres["corrclfacc"]

TRcorr_res_df["subgroup"] = ["Generalizer" if x in generalizers else "nonGeneralizer" for x in TRcorr_res_df["subid"]]
TEcorr_res_df["subgroup"] = ["Generalizer" if x in generalizers else "nonGeneralizer" for x in TEcorr_res_df["subid"]]
cross_corr_res_df["subgroup"] = ["Generalizer" if x in generalizers else "nonGeneralizer" for x in cross_corr_res_df["subid"]]

import seaborn as sns
plot_df = TRcorr_res_df.copy()
plot_df["region"] = [x.split("_")[0] for x in plot_df["roi"]]
plot_df["subgroup"] = pd.Categorical(
    ["Generalizer" if x in generalizers else "nonGeneralizer" for x in plot_df["subid"]],
    categories=["Generalizer","nonGeneralizer"],ordered=True
)
gs = sns.catplot(
    data= plot_df,
    x="region",
    y="tssim2loc",
    hue="subgroup", palette="Set1",
    kind="violin",inner="box",
    split=True, gap=0.3,fill=False,
    aspect=2

)
gs.map_dataframe(sns.stripplot, jitter=0.1, dodge=True, 
                 x="region", y="tssim2loc", hue="subgroup", alpha=1, palette="Set1")


gs = sns.catplot(
    data= plot_df,
    x="region",
    y="tssim2euc",
    hue="subgroup", palette="Set1",
    kind="violin",inner="box",
    split=True, gap=0.3,fill=False,
    aspect=2

)
gs.map_dataframe(sns.stripplot, jitter=0.1, dodge=True, 
                 x="region", y="tssim2euc", hue="subgroup", alpha=1, palette="Set1")



plot_df = TEcorr_res_df.copy()
plot_df["region"] = [x.split("_")[0] for x in plot_df["roi"]]
plot_df["subgroup"] = pd.Categorical(
    ["Generalizer" if x in generalizers else "nonGeneralizer" for x in plot_df["subid"]],
    categories=["Generalizer","nonGeneralizer"],ordered=True
)
plot_df["tarvar"] = pd.Categorical(
    plot_df["tarvar"] ,
    categories=["stim_x","stim_y"],ordered=True
)
gs = sns.catplot(
    data= plot_df,
    x="region",
    y="tssim2euc",
    col="subgroup",
    hue="tarvar", palette="Set1",
    kind="violin",inner="box",
    split=True, gap=0.3,fill=False,
    aspect=2

)
gs.map_dataframe(sns.stripplot, jitter=0.1, dodge=True, 
                 x="region", y="tssim2euc", hue="tarvar", alpha=1, palette="Set1")
gs.refline(y=0)


gs = sns.catplot(
    data= plot_df,
    x="region",
    y="tssim2loc",
    col="subgroup",
    hue="tarvar", palette="Set1",
    kind="violin",inner="box",
    split=True, gap=0.3,fill=False,
    aspect=2

)
gs.map_dataframe(sns.stripplot, jitter=0.1, dodge=True, 
                 x="region", y="tssim2euc", hue="tarvar", alpha=1, palette="Set1")
gs.refline(y=0)


plot_df = cross_corr_res_df[cross_corr_res_df.tarvar!="training"].copy()
plot_df["region"] = [x.split("_")[0] for x in plot_df["roi"]]
plot_df["subgroup"] = pd.Categorical(
    ["Generalizer" if x in generalizers else "nonGeneralizer" for x in plot_df["subid"]],
    categories=["Generalizer","nonGeneralizer"],ordered=True
)
plot_df["tarvar"] = pd.Categorical(
    plot_df["tarvar"] ,
    categories=["stim_x","stim_y"],ordered=True
)
gs = sns.catplot(
    data= plot_df,
    x="region",
    y="corrclfacc",
    col="subgroup",
    hue="tarvar", palette="Set1",
    kind="violin",inner="box",
    split=True, gap=0.3,fill=False,
    aspect=2

)
gs.map_dataframe(sns.stripplot, jitter=0.1, dodge=True, 
                 x="region", y="corrclfacc", hue="tarvar", alpha=1, palette="Set1")
gs.refline(y=0.25)