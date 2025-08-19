"""
This file run the feature decoding analysis by training the feature decoder in training/test stimuli and evaluate the decoder performance in location representation in the localizer task.

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

from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, LeavePGroupsOut,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,r2_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
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
rois = ["testgtlocParietalSup_bilateral"]#,"HPC_bilateral","vmPFC_bilateral"]#,"V1_bilateral"] # 

preprox_fun = lambda x,sess: extract_pc(concat_data([scale_feature(scale_feature(sx,2),1) for sx in split_data(x,sess)])) #scale_feature(x,1)#scale_feature(x,1)#concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #extract_pc(scale_feature(x,1)) #

# For LR
baseclf_kwargs = {'max_iter':100000,'solver':'lbfgs','multi_class':'multinomial','random_state':0}
gridsearcg_paramgrid={'C':np.geomspace(10**-5,50,num=20)} #'tol':np.logspace(-5,-3,num=3)
# For Ridge
baseclf_kwargs = {'max_iter':100000,'random_state':0} # 'solver':'lbfgs',
gridsearcg_paramgrid={'alpha':np.linspace(10**-3,50,num=20)} #'tol':np.logspace(-5,-3,num=3)
# For LDA
baseclf_kwargs = {'solver':'svd'}

# For saving
save_fn = "noncenter_navi2loc_LDAdecoding_acc_skf"
save_dir = os.path.join(ROIRSAdir,"navi2loc_decoding_results")
checkdir(save_dir)
flag_saveGSres = True
flag_saveCM = True

# Parallel jobs
njobs = 15    

res_dfs = []
confusion_mats = {}
GSres = []

n_rands = 4 # to make sure the results are not dependent on the random state
for ir, roi in enumerate(rois):
    confusion_mats[roi] = {}

    print(f"\n{ir+1}/{len(rois)}",end="\n")

    #baseclf_kwargs["random_state"] = rands
    confusion_mats[roi] = {}

    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)
        confusion_mats[roi][subid] = {}

        navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
        training_filter = subdata["stimdf"].stim_group.to_numpy() == 1
        test_filter = subdata["stimdf"].stim_group.to_numpy() == 0
        non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
        lzer_filter = subdata["stimdf"].taskname.to_numpy() == "localizer"
        filter_x = subdata["stimdf"].stim_x.to_numpy() != 0
        filter_y = subdata["stimdf"].stim_y.to_numpy() != 0        

        whitenedX = preprox_fun(subdata["preprocX"],subdata["stimdf"].stim_session.to_numpy())
        
        for tarvar in ["stim_x","stim_y"]:

            fit_accs, eval_accs = [],[]
            confusion_mats[roi][subid][tarvar] = []  
                
            for ana,fil in zip(["train2loc","test2loc"],[training_filter,test_filter]):
                # get splitter
                if tarvar=="stim_x":
                    fit_fil = np.vstack([navi_filter,fil,non_center_filter,filter_x]).all(axis=0)
                    eval_fil = np.vstack([lzer_filter,non_center_filter,filter_x]).all(axis=0)
                elif tarvar=="stim_y":
                    fit_fil = np.vstack([navi_filter,fil,non_center_filter,filter_y]).all(axis=0)
                    eval_fil = np.vstack([lzer_filter,non_center_filter,filter_y]).all(axis=0)


                # convert target labels into integers        
                target_labels = (subdata["stimdf"][tarvar].values*2+2).astype(int)
                session_labels   = subdata["stimdf"].stim_session.to_numpy()

                # get fit and evaluationa data
                fit_X, eval_X = whitenedX[fit_fil], whitenedX[eval_fil]
                fit_target, eval_target = target_labels[fit_fil], target_labels[eval_fil]
                fit_sess_labels = session_labels[fit_fil]
                
                # grid search for best hyperparameters in the fit set                
                n_splits, n_repeats = 4, 5                
                clf = LinearDiscriminantAnalysis(**baseclf_kwargs)
                clf.fit(fit_X,fit_target) # ,groups=fit_sess_labels
                #GSres.append(pd.DataFrame(clf.cv_results_).assign(roi=roi,subid=subid,target=tarvar,analysis=ana,random_state=rands))

                fit_acc = clf.score(fit_X,fit_target)
                eval_acc = clf.score(eval_X,eval_target)
                eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                confusion_mats[roi][subid][tarvar].append(eval_cfm)
                fit_accs.append(fit_acc)
                eval_accs.append(eval_acc)
            tmpdf = pd.DataFrame()
            tmpdf["fit_acc"] = fit_accs
            tmpdf["eval_acc"] = eval_accs
            tmpdf["analysis"] = ["train2loc","test2loc"]
            res_dfs.append(
                tmpdf.assign(roi=roi,
                        subid=subid,
                        target=tarvar
                        )
                            )
    # save roi results
    roires_df = pd.concat(res_dfs).reset_index(drop=True)
    decoder = {
        "estimator":clf.__str__(), # get the name of the decoder class used
        "baseclf_kwargs":baseclf_kwargs
        }
    roi_res_dict = {
        "decoder":decoder,
        "performance":roires_df[roires_df.roi==roi].copy().reset_index(drop=True)
        }
    if flag_saveCM:
        roi_res_dict["confusion_matrices"] = confusion_mats[roi]
    dump(roi_res_dict, os.path.join(save_dir,f"{save_fn}_{roi}.pkl"))

# save all the results
res_df = pd.concat(res_dfs).reset_index(drop=True)
decoder = {"estimator":clf.__str__(),
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
    "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(ROIRSAdir,f"{save_fn}.pkl"))