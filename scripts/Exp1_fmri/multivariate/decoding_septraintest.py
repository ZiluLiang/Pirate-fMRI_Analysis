"""
This file run the feature decoding analysis by training the feature decoder in training stimuli and evaluate the decoder performance in test stimuli.

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
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, LeavePGroupsOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix,precision_score, recall_score

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
rois =  [x for x in list(roi_data.keys()) if "bilateral" in x] + ["allgtlocPrecentral_left"]


cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

SCORER_DICT = {"acc":accuracy_score,
               "f1micro":lambda y_true,y_pred: f1_score(y_true,y_pred,average="micro"),
               "precision":lambda y_true,y_pred: precision_score(y_true,y_pred,average="micro"),
               "recall":lambda y_true,y_pred: recall_score(y_true,y_pred,average="micro")}


preprox_fun = lambda x,sess: concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #scale_feature(x,1)#scale_feature(x,1)#concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #extract_pc(scale_feature(x,1)) #

# For LR
baseclf_kwargs = {'max_iter':100000,'solver':'lbfgs','class_weight':'balanced','random_state':100}
gridsearcg_paramgrid={'C':np.concatenate([np.geomspace(10**-5,5,num=10),np.linspace(5,100,num=10)])}


# For saving
save_fn = "navi_septraintest_LRdecoding_acc"
save_dir = os.path.join(ROIRSAdir,"navitraintestsplit_decoding_results")
checkdir(save_dir)
flag_saveGSres = True
flag_saveCM = True
n_jobs = 10

n_rands = 20 # to make sure the results are not dependent on the random state

res_dfs = []
confusion_mats = {}
GSres = []
for ir,roi in enumerate([x for x in rois if "bilateral" in x] + ["allgtlocPrecentral_left"]):
    confusion_mats[roi] = {}
    for rands in range(n_rands):
        print(f"\n{ir+1}/{len(rois)} - randomstates: {rands}/{n_rands}",end="\n")  
        baseclf_kwargs["random_state"] = rands
        confusion_mats[roi][rands] = {}
        
        for subdata,subid in zip(roi_data[roi],subid_list):
            print(f"{roi} - {subid}",end="\r",flush=True)
            confusion_mats[roi][rands][subid] = {}
            stimdf = subdata["stimdf"].copy()
            navi_filter = stimdf.taskname.to_numpy() == "navigation"
            
            tarvar = "stim_group"
            confusion_mats[roi][rands][subid][tarvar] = []
            fil = navi_filter
            whitenedX = subdata["preprocX"][fil]
            session_labels   = stimdf[fil].stim_session.to_numpy()
            preproced_X = preprox_fun(whitenedX,session_labels)

            # convert target labels into integers        
            target_labels = stimdf[tarvar].values[fil]

            # fit/evaluation group splitter:
                # we are using stratified kfold here bc the number of train vs test is unbalanced and our dataset is small
                # to make sure that we have enough data in the fitting bit to do gridsearch, we do 4-fold cv
                # to make sure that we don't find a very specific decision boundary between train-test due to the random split
            n_splits, n_repeats = 4, 5
            FIT_SCORES = dict(zip(SCORER_DICT.keys(),[[] for _ in range(len(SCORER_DICT))]))
            EVAL_SCORES = dict(zip(SCORER_DICT.keys(),[[] for _ in range(len(SCORER_DICT))]))
            skf = StratifiedKFold(n_splits=n_splits,random_state=rands,shuffle=True)
            for i, (fit_idx, eval_idx) in enumerate(skf.split(preproced_X,target_labels)):
                # split data in to fit and evaluation set
                fit_sess_labels = session_labels[fit_idx]
                fit_X, fit_target = preproced_X[fit_idx], target_labels[fit_idx] 
                eval_X, eval_target = preproced_X[eval_idx], target_labels[eval_idx]

                clf = GridSearchCV(LogisticRegression(**baseclf_kwargs),
                                param_grid=gridsearcg_paramgrid,
                                cv=RepeatedStratifiedKFold(n_repeats=n_repeats,n_splits=n_splits-1,random_state=rands),
                                refit=True,
                                n_jobs=15)
                clf.fit(fit_X,fit_target)
                GSres.append(
                    pd.DataFrame(clf.cv_results_).assign(
                    roi=roi,subid=subid,target=tarvar,straifiedfold=i
                    )
                    )

                for k,scorer in SCORER_DICT.items():
                    FIT_SCORES[k].append(scorer(fit_target,clf.predict(fit_X)))
                    EVAL_SCORES[k].append(scorer(eval_target,clf.predict(eval_X)))

                eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                confusion_mats[roi][rands][subid][tarvar].append(eval_cfm)
            
            tmpdf = pd.DataFrame()
            for k in SCORER_DICT.keys():
                tmpdf[f"fit_{k}"] = FIT_SCORES[k]
                tmpdf[f"eval_{k}"] = EVAL_SCORES[k]
            tmpdf["straifiedfold"] = np.arange(n_splits)
            res_dfs.append(
                tmpdf.assign(roi=roi,
                        subid=subid,
                        target=tarvar
                        )
                        )
        roires_df = pd.concat(res_dfs).reset_index(drop=True)
        roiGSres_df = pd.concat(GSres).reset_index(drop=True)
        decoder = {"estimator":clf.estimator.__str__(),
                "gsparamgrid":gridsearcg_paramgrid,
                "baseclf_kwargs":baseclf_kwargs}

        dump({#"preprocessingfun":preprox_fun,
            "GSresults":roiGSres_df[roiGSres_df.roi==roi].copy().reset_index(drop=True),
            "decoder":decoder,
            "performance":roires_df[roires_df.roi==roi].copy().reset_index(drop=True),
            "confusion_matrices":confusion_mats[roi]},
            os.path.join(save_dir,f"{save_fn}_{roi}.pkl"))
res_df = pd.concat(res_dfs).reset_index(drop=True)
GSres_df = pd.concat(GSres).reset_index(drop=True)
decoder = {"estimator":clf.estimator.__str__(),
           "gsparamgrid":gridsearcg_paramgrid,
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
    "GSresults":GSres_df,
    "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(save_dir,f"{save_fn}.pkl"))