"""
This file run the feature decoding analysis by training the feature decoder in training/test stimuli and evaluate the decoder performance in pirate positions representation in the localizer task.

During decoder fitting, hyperparameter C is optimized using a grid search with 25 values between 10^-5 and 50 using a gridsearchCV approach
During this hyperparameter search, the fitting data is again processed through repeated stratified k-fold fashion to look for hyperparameter that maximize the decoding accuracy in the validation set.

before running the decoding analysis, wew aligned each participants' position representation to that in the training location representation with orthogonal procrustes rotation.   
to make sure this does not yield any bias, we also shuffled the labels of position representation, then re-did the rotation analysis, and then did the decoding analysis.


"""


import itertools
import numpy as np
import seaborn as sns

import json
import time
import pandas as pd
import glob
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.MVPA.preprocessors import scale_feature, average_odd_even_session,normalise_multivariate_noise, split_data, concat_data,kabsch_algorithm
from zpyhelper.filesys import checkdir

from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, LeavePGroupsOut,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,r2_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes

import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore', category=FutureWarning)


study_scripts   = r"D:\OneDrive - Nexus365\pirate_ongoing\scripts\Exp1_fmri"
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]

print("N_cohort 1: ",len(cohort1ids), "  N_cohort 2: ",len(cohort2ids), "N_Total: ",len(subid_list))

ROIRSAdir = r"D:\OneDrive - Nexus365\pirate_ongoing\AALandHCPMMP1andFUNCcluster"
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois =  [x for x in list(roi_data.keys()) if "bilateral" in x] + ["allgtlocPrecentral_left"]
rois = ["V1_bilateral","testgtlocParietalSup_bilateral","HPC_bilateral","vmPFC_bilateral"]
rois = rois
#rois = ["V1_bilateral","HPC_bilateral","vmPFC_bilateral","testgtlocParietalSup_bilateral"]
preprox_fun = lambda x,sess: concat_data([scale_feature(scale_feature(sx,2),1) for sx in split_data(x,sess)]) #scale_feature(x,1)#scale_feature(x,1)#concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #extract_pc(scale_feature(x,1)) #

# For LR
baseclf_kwargs = {'max_iter':100000,'solver':'lbfgs','multi_class':'multinomial','random_state':0}
gridsearcg_paramgrid={'C':np.concatenate([np.geomspace(10**-5,5,num=15),np.linspace(5,50,num=10)]),
                      'tol':np.geomspace(10**-7,5,num=5)}

# For saving
#save_fn = "flzeralignedprocrustes"
save_fn = "noncenter_navi2loc_LRdecoding_acc_skf"
save_dir = os.path.join(ROIRSAdir,"navi2loc_decoding_results")#"rotation_testres") # navi2loc_decoding_results
checkdir(save_dir)
flag_saveGSres = True
flag_saveCM = True

# Parallel jobs
njobs = 15    

res_dfs = []
confusion_mats = {}
GSres = []

n_rands = 20 # to make sure the results are not dependent on the random state
for ir, roi in enumerate(rois):
    confusion_mats[roi] = {}

    for rands in range(n_rands):
        print(f"\n{ir+1}/{len(rois)} - randomstates: {rands}/{n_rands}",end="\n")

        baseclf_kwargs["random_state"] = rands
        confusion_mats[roi][rands] = {}

        for subdata,subid in zip(roi_data[roi],subid_list):
            print(f"{roi} - {subid}",end="\r",flush=True)
            confusion_mats[roi][rands][subid] = {}
            # get non-center stimuli
            non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
            center_filter = [all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
            ncXs, ncDF = subdata["preprocX"][non_center_filter], subdata["stimdf"][non_center_filter].copy().reset_index(drop=True)
            cXs, cDF = subdata["preprocX"][center_filter], subdata["stimdf"][center_filter].copy().reset_index(drop=True)

            # get filters            
            navi_filter = ncDF.taskname.to_numpy() == "navigation"
            training_filter = ncDF.stim_group.to_numpy() == 1
            test_filter = ncDF.stim_group.to_numpy() == 0
            lzer_filter = ncDF.taskname.to_numpy() == "localizer"
            filter_x = ncDF.stim_x.to_numpy() != 0
            filter_y = ncDF.stim_y.to_numpy() != 0        

            whitenedX = preprox_fun(ncXs,ncDF.stim_session.to_numpy())
            lzerX = whitenedX[lzer_filter,:]
                
            # convert target labels into integers        
            session_labels   = ncDF.stim_session.to_numpy()

            for tarvar in ["stim_x","stim_y"]:
                # get splitter
                if tarvar=="stim_x":
                    tr_fil = np.vstack([navi_filter,training_filter,filter_y]).all(axis=0)
                    lz_fil = np.vstack([lzer_filter,filter_y]).all(axis=0)
                elif tarvar=="stim_y":
                    tr_fil = np.vstack([navi_filter,training_filter,filter_x]).all(axis=0)
                    lz_fil = np.vstack([lzer_filter,filter_x]).all(axis=0)
                
                trX = np.vstack(
                    [np.mean(split_data(whitenedX[tr_fil],session_labels[tr_fil]),axis=0),
                     np.mean(cXs[cDF.taskname.to_numpy() == "navigation"],axis=0)]
                )
                lzX = np.vstack(
                    [whitenedX[lz_fil,:],
                     cXs[cDF.taskname.to_numpy() == "localizer"]
                     ]
                )
                
                #RlzX, RmatO,_ = kabsch_algorithm(lzX, trX)
                #RlzerX = lzerX@Rmat

                # shuffle within each row independently?
                #rng = np.random.default_rng(rands)
                #shuffled = []
                #for x in lzerX:
                #    shuffled.append(rng.permutation(x))
                #lzX_rowshuffled = np.vstack(shuffled)
                #RlzX_rowshuffled, RmatRS, _ = kabsch_algorithm(lzX_rowshuffled[:lzX.shape[0],:], trX)

                #lzX_tarshuffled = rng.permutation(lzerX)
                #RlzX_tarshuffled, RmatTS, _ = kabsch_algorithm(lzX_tarshuffled[:lzX.shape[0],:], trX)

                #randommatrix = rng.random(lzerX.shape)
                #Rrandmat, RmatRAND, _ = kabsch_algorithm(randommatrix[:lzX.shape[0],:], trX)

                dataset = {
                    "original":              whitenedX#np.vstack([whitenedX[~lzer_filter], lzerX]),
                    #"rotated":               np.vstack([whitenedX[~lzer_filter], lzerX@RmatO]),
                    #"rowindshuffledrotated": np.vstack([whitenedX[~lzer_filter], lzX_rowshuffled@RmatRS]),
                    #"tarshuffledrotated":    np.vstack([whitenedX[~lzer_filter], lzX_tarshuffled@RmatO]),
                    #"randommatrixrotated":   np.vstack([whitenedX[~lzer_filter], randommatrix@RmatO])
                }
            
                for dsname, whitenedX in dataset.items():
                    target_labels = (ncDF[tarvar].values*2+2).astype(int)
                
                    fit_accs, eval_accs = [],[]
                    confusion_mats[roi][rands][subid][tarvar] = []  
                    
                    for ana,fil in zip(["train2loc","test2loc"],[training_filter,test_filter]):
                        # get splitter
                        if tarvar=="stim_x":
                            fit_fil = np.vstack([navi_filter,fil,filter_x]).all(axis=0)
                            eval_fil = np.vstack([lzer_filter,filter_x]).all(axis=0)
                        elif tarvar=="stim_y":
                            fit_fil = np.vstack([navi_filter,fil,filter_y]).all(axis=0)
                            eval_fil = np.vstack([lzer_filter,filter_y]).all(axis=0)


                        
                        # get fit and evaluationa data
                        fit_X, eval_X = whitenedX[fit_fil], whitenedX[eval_fil]
                        fit_target, eval_target = target_labels[fit_fil], target_labels[eval_fil]
                        fit_sess_labels = session_labels[fit_fil]

                        # grid search for best hyperparameters in the fit set                
                        n_splits, n_repeats = 4, 10                
                        clf = GridSearchCV(LogisticRegression(**baseclf_kwargs),
                                        param_grid=gridsearcg_paramgrid,
                                        cv=RepeatedStratifiedKFold(n_repeats=n_repeats,n_splits=n_splits,random_state=rands),
                                        n_jobs=njobs)
                        clf.fit(fit_X,fit_target)
                        GSres.append(pd.DataFrame(clf.cv_results_).assign(roi=roi,subid=subid,target=tarvar,analysis=ana,random_state=rands,dataset=dsname))

                        fit_acc = clf.score(fit_X,fit_target)
                        eval_acc = clf.score(eval_X,eval_target)
                        eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                        confusion_mats[roi][rands][subid][tarvar].append(eval_cfm)
                        fit_accs.append(fit_acc)
                        eval_accs.append(eval_acc)
                    tmpdf = pd.DataFrame()
                    tmpdf["fit_acc"] = fit_accs
                    tmpdf["eval_acc"] = eval_accs
                    tmpdf["analysis"] = ["train2loc","test2loc"] # 
                    res_dfs.append(
                        tmpdf.assign(roi=roi,
                                subid=subid,
                                target=tarvar,
                                random_state=rands,
                                dataset=dsname
                                )
                                    )
    # save roi results
    roires_df = pd.concat(res_dfs).reset_index(drop=True)
    roiGSres_df = pd.concat(GSres).reset_index(drop=True)
    decoder = {
        "estimator":clf.estimator.__str__(), # get the name of the decoder class used
        "gsparamgrid":gridsearcg_paramgrid,
        "baseclf_kwargs":baseclf_kwargs
        }
    roi_res_dict = {
        "decoder":decoder,
        "performance":roires_df[roires_df.roi==roi].copy().reset_index(drop=True)
        }
    if flag_saveCM:
        roi_res_dict["confusion_matrices"] = confusion_mats[roi]
    if flag_saveGSres:
        roi_res_dict["GSresults"] = roiGSres_df[roiGSres_df.roi==roi].copy().reset_index(drop=True)
    dump(roi_res_dict, os.path.join(save_dir,f"{save_fn}_{roi}.pkl"))

# save all the results
res_df = pd.concat(res_dfs).reset_index(drop=True)
GSres_df = pd.concat(GSres).reset_index(drop=True)
decoder = {"estimator":clf.estimator.__str__(),
           "gsparamgrid":gridsearcg_paramgrid,
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
    "GSresults":GSres_df,
    "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(save_dir,f"{save_fn}.pkl"))

# # ## to check results
nclocnavi_decoding_res = load(os.path.join(save_dir,f"{save_fn}.pkl"))
print(nclocnavi_decoding_res.keys())
print(nclocnavi_decoding_res['performance'].columns)
print(nclocnavi_decoding_res['performance'].target.unique())

#check average decoding accuracy of all conditions
nclocnavidc_performance_df = nclocnavi_decoding_res["performance"].copy()
nclocnavidc_performance_dfsum = nclocnavidc_performance_df.groupby(["dataset","roi","subid","analysis","target"])[["fit_acc","eval_acc"]].mean().reset_index()
nclocnavidc_performance_dfsum["subgroup"] = pd.Categorical(
    ["Generalizer" if subid in generalizers else "nonGeneralizer" for subid in nclocnavidc_performance_dfsum["subid"]],
    categories=["Generalizer","nonGeneralizer"],ordered=True
)
nclocnavidc_performance_dfsum["target"] = pd.Categorical(nclocnavidc_performance_dfsum["target"],categories=["stim_x","stim_y"],ordered=True)
nclocnavidc_performance_dfsum[['region','side']] = nclocnavidc_performance_dfsum.roi.str.split("_",expand=True)

nclocnavidc_performance_df["subgroup"] = ["Generalizer" if subid in generalizers else "nonGeneralizer" for subid in nclocnavidc_performance_df["subid"]]
nclocnavidc_performance_df[['region','side']] = nclocnavidc_performance_df.roi.str.split("_",expand=True)

gs = sns.catplot(data=nclocnavidc_performance_dfsum,
            x="dataset",y="eval_acc",
            hue="subgroup",palette="tab10",
            col="target",row="analysis",
            kind="bar",errorbar="se",alpha=0.5,aspect=2)
gs.map_dataframe(sns.stripplot,
                 x="dataset",y="eval_acc",
                hue="subgroup",palette="tab10",
                dodge=True)
for ax in gs.axes.flatten():
    ax.set_xlabel("Decoding Target",fontdict={"fontsize":12,"fontweight":"bold"},visible=True)
    ax.set_ylabel("Decoding Accuracy",fontdict={"fontsize":12,"fontweight":"bold"})
    ax.axhline(0.25,linestyle="--",color="black")

gs = sns.catplot(data=nclocnavidc_performance_dfsum,
            x="dataset",y="fit_acc",
            hue="subgroup",palette="tab10",
            col="target",row="analysis",
            kind="bar",errorbar="se",alpha=0.5,aspect=2)
for ax in gs.axes.flatten():
    ax.set_xlabel("Decoding Target",fontdict={"fontsize":12,"fontweight":"bold"},visible=True)
    ax.set_ylabel("(Fitting) Decoding Accuracy",fontdict={"fontsize":12,"fontweight":"bold"})
    ax.axhline(0.25,linestyle="--",color="black")

gs.map_dataframe(sns.stripplot,
                 x="dataset",y="fit_acc",
                hue="subgroup",palette="tab10",
                dodge=True)