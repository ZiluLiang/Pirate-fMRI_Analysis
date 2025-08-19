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

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import compute_rdm,lower_tri,upper_tri, compute_rdm_nomial, compute_rdm_identity
from zpyhelper.MVPA.preprocessors import scale_feature, average_odd_even_session,normalise_multivariate_noise, split_data, concat_data,extract_pc,average_flexi_session

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM
from scripts.Exp1_fmri.multivariate.mvpa_runner import MVPARunner
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim, minmax_scale, generate_filters
from src.utils.composition_modelfit import multi_start_optimisation

import sklearn
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, LeavePGroupsOut
from sklearn.metrics import accuracy_score,r2_score, confusion_matrix

import scipy

import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore', category=FutureWarning)


study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
studydata_dir  = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]

print("N_cohort 1: ",len(cohort1ids), "  N_cohort 2: ",len(cohort2ids), "N_Total: ",len(subid_list))

fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
fmribeh_dir = os.path.join(fmridata_dir,'beh')

ROIRSAdir = os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1andFUNCcluster')
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois =  list(roi_data.keys())


cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

preprox_fun = lambda x,sess: concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #scale_feature(x,1)#scale_feature(x,1)#concat_data([scale_feature(sx,1) for sx in split_data(x,sess)]) #extract_pc(scale_feature(x,1)) #
# For SVC
#baseclf_kwargs = {'max_iter':-1,'kernel':'linear','random_state':0}
#gridsearcg_paramgrid={'C':np.linspace(10**-10,10,num=50)} #'tol':np.logspace(-5,-3,num=5),

# For LR
baseclf_kwargs = {'max_iter':100000,'solver':'lbfgs','multi_class':'multinomial','random_state':0}
gridsearcg_paramgrid={'C':np.geomspace(10**-10,100,num=1000)} #'tol':np.logspace(-5,-3,num=3)
        
res_dfs = []
confusion_mats = {}
GSres = []
for ir, roi in enumerate(rois):
    print(f"{ir+1}/{len(rois)}",end="\n")
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
        
        for ana,fil in zip(["train2loc","test2loc"],[training_filter,test_filter]):
            confusion_mats[roi][subid][ana] = []
            fit_accs, eval_accs = [],[]
            for tarvar in ["stim_x","stim_y"]:
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
                clf = GridSearchCV(LogisticRegression(**baseclf_kwargs),
                                param_grid=gridsearcg_paramgrid,
                                cv=LeaveOneGroupOut(),
                                n_jobs=15)
                clf.fit(fit_X,fit_target,groups=fit_sess_labels) #,groups=fit_session_labels
                GSres.append(pd.DataFrame(clf.cv_results_).assign(roi=roi,subid=subid,target=tarvar,analysis=ana))

                fit_acc = clf.score(fit_X,fit_target)
                eval_acc = clf.score(eval_X,eval_target)
                eval_cfm = confusion_matrix(eval_target,clf.predict(eval_X))

                confusion_mats[roi][subid][ana].append(eval_cfm)
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
res_df = pd.concat(res_dfs).reset_index(drop=True)
GSres_df = pd.concat(GSres).reset_index(drop=True)
decoder = {"estimator":clf.estimator.__str__(),
           "gsparamgrid":gridsearcg_paramgrid,
           "baseclf_kwargs":baseclf_kwargs}

dump({#"preprocessingfun":preprox_fun,
    "GSresults":GSres_df,
    "decoder":decoder,"performance":res_df,"confusion_matrices":confusion_mats},
     os.path.join(ROIRSAdir,"noncenter_navi2loc_LRdecoding_acc.pkl"))