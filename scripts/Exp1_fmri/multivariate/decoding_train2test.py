"""
This file run the feature decoding analysis by training the feature decoder in training stimuli and evaluate the decoder performance in test stimuli.

During decoder fitting, hyperparameter C is optimized using a stratified k-fold gridsearchCV approach.
During this hyperparameter search, the fitting data is processed to look for hyperparameter that maximize the decoding accuracy in the validation set (a subset of the decoder fitting dataset).

"""

import numpy as np
import json
import time
import pandas as pd
import glob
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.MVPA.preprocessors import scale_feature, split_data, concat_data,extract_pc
from zpyhelper.filesys import checkdir

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, RepeatedStratifiedKFold 
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix,precision_score, recall_score

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')

sys.path.append(project_path)

with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]

print("N_Total: ",len(subid_list))

roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois =  list(roi_data.keys())

SCORER_DICT = {"acc":accuracy_score,
               "f1micro":lambda y_true,y_pred: f1_score(y_true,y_pred,average="micro"),
               "precision":lambda y_true,y_pred: precision_score(y_true,y_pred,average="micro"),
               "recall":lambda y_true,y_pred: recall_score(y_true,y_pred,average="micro")}

preprox_fun = lambda x,sess: concat_data([scale_feature(sx,2) for sx in split_data(x,sess)]) 

# For LR
baseclf_kwargs = {'max_iter':100000,'solver':'lbfgs','multi_class':'multinomial','random_state':0}
gridsearcg_paramgrid={'C':np.concatenate([np.geomspace(10**-10,5,num=50),np.linspace(5,100,num=50)])}

# For saving
save_fn = "noncenter_train2test_LRdecoding_acc_skf"
save_dir = os.path.join(ROIRSAdir,"train2test_decoding_results")
checkdir(save_dir)
flag_saveGSres = True
flag_saveCM = True

# Parallel jobs
njobs = 15       

res_dfs = []
confusion_mats = {}
GSres = []

n_rands = 20 # to make sure the results are not dependent on the random state
for ir,roi in enumerate(rois):#rois):
    confusion_mats[roi] = {}
    for rands in range(n_rands):
        print(f"\n{ir+1}/{len(rois)} - randomstates: {rands}/{n_rands}",end="\n")  
         
        baseclf_kwargs["random_state"] = rands
        confusion_mats[roi][rands] = {}

        for subdata,subid in zip(roi_data[roi],subid_list):
            print(f"{roi} - {subid}",end="\r",flush=True)
            confusion_mats[roi][rands][subid] = {}

            navi_filter = subdata["stimdf"].taskname.to_numpy() == "navigation"
            non_center_filter = [not all([x==0, y==0]) for x,y in subdata["stimdf"][["stim_x","stim_y"]].to_numpy()]
            filter_x = subdata["stimdf"].stim_x.to_numpy() != 0
            filter_y = subdata["stimdf"].stim_y.to_numpy() != 0        
            
            for tarvar in ["stim_x","stim_y"]: # we do it separately for training and test stimuli
                confusion_mats[roi][rands][subid][tarvar] = []
                if tarvar=="stim_x":
                    fil = np.vstack([navi_filter,non_center_filter,filter_x]).all(axis=0)
                elif tarvar=="stim_y":
                    fil = np.vstack([navi_filter,non_center_filter,filter_y]).all(axis=0)
                else:
                    fil = np.vstack([navi_filter,non_center_filter]).all(axis=0)

                whitenedX = subdata["preprocX"][fil]
                session_labels   = subdata["stimdf"][fil].stim_session.to_numpy()
                stimgroup_labels = subdata["stimdf"][fil].stim_group.to_numpy()
                heldout_groups = [0,1]#0 is test, 1 is train stim
                heldout_types = ["train2test","test2train"]
                preproced_X = preprox_fun(whitenedX,session_labels)

                # convert target labels into integers        
                target_labels = (subdata["stimdf"][tarvar].values[fil]*2+2).astype(int)

                FIT_SCORES = dict(zip(SCORER_DICT.keys(),[[] for _ in range(len(SCORER_DICT))]))
                EVAL_SCORES = dict(zip(SCORER_DICT.keys(),[[] for _ in range(len(SCORER_DICT))]))

                # loop over heldout groups
                for heldout_stimgroup in heldout_groups:
                    # filter for spliting fit and evaluation (heldout) set
                    fit_fil, eval_fil = stimgroup_labels!=heldout_stimgroup, stimgroup_labels==heldout_stimgroup

                    # split data in to fit and evaluation set
                    fit_sg_labels, fit_sess_labels = stimgroup_labels[fit_fil], session_labels[fit_fil]
                    fit_X, fit_target = preproced_X[fit_fil], target_labels[fit_fil] 
                    eval_X, eval_target = preproced_X[eval_fil], target_labels[eval_fil]
                    
                    # double check if the fit set has only either training stimuli or test stimuli and heldout set has the other
                    assert np.logical_and(np.unique(fit_sg_labels).size == 1,np.unique(fit_sg_labels)[0]!=heldout_stimgroup)

                    # grid search for best hyperparameters in the fit set
                    n_splits = 4 if heldout_stimgroup==1 else 2 # 4-fold for fit on test stimuli, 2-fold for training stimuli           
                    n_repeats = 20 if heldout_stimgroup==1 else 10 # 20 repeats for fit on test stimuli, 10 repeats for training stimuli
                    clf = GridSearchCV(LogisticRegression(**baseclf_kwargs),
                                    param_grid=gridsearcg_paramgrid,
                                    cv=RepeatedStratifiedKFold(n_repeats=20,n_splits=n_splits),
                                    refit=True,
                                    n_jobs=njobs)
                    clf.fit(fit_X,fit_target) 
                    GSres.append(pd.DataFrame(clf.cv_results_).assign(roi=roi,subid=subid,target=tarvar,heldoutgroup=heldout_stimgroup,heldouttype=heldout_types[heldout_stimgroup],random_state=rands))

                    for k,scorer in SCORER_DICT.items():
                        FIT_SCORES[k].append(scorer(fit_target,clf.predict(fit_X)))
                        EVAL_SCORES[k].append(scorer(eval_target,clf.predict(eval_X)))

                    eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                    confusion_mats[roi][rands][subid][tarvar].append(eval_cfm)
                tmpdf = pd.DataFrame()
                for k in SCORER_DICT.keys():
                    tmpdf[f"fit_{k}"] = FIT_SCORES[k]
                    tmpdf[f"eval_{k}"] = EVAL_SCORES[k]
                tmpdf["heldoutgroup"] = heldout_groups
                tmpdf["heldouttype"] = [heldout_types[heldout_stimgroup] for heldout_stimgroup in heldout_groups]
                res_dfs.append(
                    tmpdf.assign(roi=roi,
                                 subid=subid,
                                 target=tarvar,
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

# save all the results
res_dfs = []
for roi in rois:    
    res_dfs.append(
        load(os.path.join(save_dir,f"{save_fn}_{roi}.pkl"))["performance"].assign(roi=roi)
    )
res_df = pd.concat(res_dfs).reset_index(drop=True)
decoder = {# for later double-checking what was done and how
    "estimator":clf.estimator.__str__(),
    "gsparamgrid":gridsearcg_paramgrid,
    "baseclf_kwargs":baseclf_kwargs
    }

final_res_dict = {
    "decoder":decoder,
    "performance":res_df
    }
if flag_saveGSres:
    GSres_df = pd.concat(GSres).reset_index(drop=True)
    final_res_dict["GSresults"] = GSres_df
if flag_saveCM:
    final_res_dict["confusion_matrices"] = confusion_mats

dump(final_res_dict, os.path.join(save_dir,f"{save_fn}.pkl"))

### check the results
import seaborn as sns
res_df = load(os.path.join(save_dir,f"{save_fn}.pkl"))["performance"]
res_df_sum = res_df.groupby(["subid","roi","target","heldouttype"]).mean().reset_index()
res_df_sum["subgroup"] = ["generalizer" if x in generalizers else "nongeneralizer" for x in res_df_sum.subid]
gs = sns.catplot(data=res_df_sum[res_df_sum.roi.str.contains("bilateral")],
            x="subgroup",y="eval_acc",hue="target",
            col="roi",row="heldouttype",
            kind="bar",errorbar="se")
for ax in gs.axes.flatten():
    ax.axhline(0.25,linestyle="--",color="black")
