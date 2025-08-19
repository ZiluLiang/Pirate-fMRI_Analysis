"""
This file run the feature decoding analysis with the training stimuli that are not the center stimuli.

The decoding is done using a logistic regression classifier with a leave-one-session-out cross-validation (fit on 3 session, eval on the 1 heldout).
During decoder fitting, hyperparameter C is optimized using a grid search with 50 values between 10^-10 and 10 using a gridsearchCV approach
During this hyperparameter search, the fitting data is again processed through leave-one-session-out fashion to look for hyperparameter that maximize the decoding accuracy in the validation set.

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

from zpyhelper.MVPA.preprocessors import scale_feature, average_odd_even_session,normalise_multivariate_noise, split_data, concat_data,extract_pc,average_flexi_session
from zpyhelper.filesys import checkdir

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.metrics import accuracy_score,r2_score, confusion_matrix

import scipy

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
rois =  [x for x in list(roi_data.keys()) if "bilateral" in x]# + ["allgtlocPrecentral_left"]
rois = ["HPC_bilateral","vmPFC_bilateral","V1_bilateral"]

preprox_fun = lambda x,sess: concat_data([scale_feature(scale_feature(sx,2),1) for sx in split_data(x,sess)]) #scale_feature(x,1)#scale_feature(x,1)#concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #extract_pc(scale_feature(x,1)) #

# For LR
baseclf_kwargs = {'max_iter':100000,'solver':'lbfgs','multi_class':'multinomial','random_state':0}
gridsearcg_paramgrid={'C':np.concatenate([np.geomspace(10**-5,5,num=15),np.linspace(5,100,num=15)])}

# For saving
save_fn = "allstimidendity_LRdecoding_acc_skf"
save_dir = os.path.join(ROIRSAdir,"TSstimidentity_decoding_results")
checkdir(save_dir)
flag_saveGSres = True
flag_saveCM = True
n_jobs = 10

res_dfs = []
confusion_mats = {}
GSres = []

n_rands = 20 # to make sure the results are not dependent on the random state
for ir,roi in enumerate(rois):
    confusion_mats[roi] = {}
    for rands in range(n_rands):
        print(f"\n{ir+1}/{len(rois)} - randomstates: {rands}/{n_rands}",end="\n")  
         
        baseclf_kwargs["random_state"] = rands
        confusion_mats[roi][rands] = {}

        for subdata,subid in zip(roi_data[roi],subid_list):
            print(f"{ir+1}/{len(rois)}: {roi} - {subid}",end="\r",flush=True)
            confusion_mats[roi][rands][subid] = {}

            navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
            test_fileter = np.vstack([subdata["stimdf"].stim_group.to_numpy() == 0,
                                      navi_filter]).all(axis=0)
            
            tarvar = "stim_id"
            
            confusion_mats[roi][rands][subid] = []
            for stimsubset, fil in zip(["all","test"],[navi_filter,test_fileter]):                

                whitenedX = subdata["preprocX"][fil]
                session_labels = subdata["stimdf"][fil].stim_session.to_numpy()
                preproced_X = preprox_fun(whitenedX,session_labels)
            
                
                target_labels = subdata["stimdf"][tarvar].to_numpy()[fil]
                
                # fit/evaluation group splitter:
                # we are using stratified kfold here bc our dataset is small and to avoid overfitting to run-to-run variance
                # to make sure that we have enough data in the fitting bit to do gridsearch, we do 4-fold cv
                n_splits = 4
                fit_accs, eval_accs = [],[]
                skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=rands)
                for i, (fit_idx, eval_idx) in enumerate(skf.split(preproced_X,target_labels)):
                #for heldout_sessions in np.unique(session_labels):
                    
                    #fit_session_labels = session_labels[session_labels!=heldout_sessions]
                    #fit_X, fit_target = preproced_X[session_labels!=heldout_sessions], target_labels[session_labels!=heldout_sessions] 
                    #eval_X, eval_target = preproced_X[session_labels==heldout_sessions], target_labels[session_labels==heldout_sessions]

                    # split data in to fit and evaluation set
                    fit_sess_labels = session_labels[fit_idx]
                    fit_X, fit_target = preproced_X[fit_idx], target_labels[fit_idx] 
                    eval_X, eval_target = preproced_X[eval_idx], target_labels[eval_idx]
                    
                    
                    # to make sure that we don't find a very specific decision boundary due to the random split
                    # we use repeated stratified kfold with 5 repeats
                    clf = GridSearchCV(LogisticRegression(**baseclf_kwargs),
                                    param_grid=gridsearcg_paramgrid,
                                    cv=RepeatedStratifiedKFold(n_repeats=5,n_splits=n_splits-1),
                                    n_jobs=n_jobs)
                    clf.fit(fit_X,fit_target)
                    GSres.append(pd.DataFrame(clf.cv_results_).assign(roi=roi,subid=subid,target=tarvar,straifiedfold=i,stimsubset=stimsubset))#heldoutrun=heldout_sessions))

                    fit_acc = clf.score(fit_X,fit_target)
                    eval_acc = clf.score(eval_X,eval_target)
                    eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                    confusion_mats[roi][rands][subid].append(eval_cfm)
                    fit_accs.append(fit_acc)
                    eval_accs.append(eval_acc)
                tmpdf = pd.DataFrame()
                tmpdf["fit_acc"] = fit_accs
                tmpdf["eval_acc"] = eval_accs
                tmpdf["straifiedfold"] = np.arange(i+1)
                res_dfs.append(
                    tmpdf.assign(roi=roi,
                            subid=subid,
                            target=tarvar,
                            stimsubset=stimsubset,
                            random_state=rands)
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


res_df = pd.concat(res_dfs).reset_index(drop=True)
GSres_df = pd.concat(GSres).reset_index(drop=True)
decoder = {"estimator":clf.estimator.__str__(),
           "gsparamgrid":gridsearcg_paramgrid,
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
    "GSresults":GSres_df,
    "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(save_dir,f"{save_fn}.pkl"))

