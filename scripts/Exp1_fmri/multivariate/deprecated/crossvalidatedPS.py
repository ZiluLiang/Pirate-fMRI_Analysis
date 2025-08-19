import itertools
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import json
from copy import deepcopy
import os
import time
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature,average_odd_even_session
from zpyhelper.MVPA.estimators import MultipleRDMRegression

project_path = r'E:\pirate_fmri\Analysis'
import sys
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim,generate_filters
from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Load data
project_path  = r'E:\pirate_fmri\Analysis'
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
studydata_dir = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")
print(f"N_cohort1 = {len(cohort1ids)}")
print(f"N_cohort2 = {len(cohort2ids)}")

cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

base_rois = ["HPC","vmPFC","V1","V2"]
rois =  [f"{x}_bilateral" for x in base_rois]
ROIRSAdir = os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1')
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))

## select activity pattern of non-center training stimuli
sub_patterns = {}
sub_stimdfs = {}
for roi in rois:
    sub_patterns[roi], sub_stimdfs[roi] = [],[]
    for subid,subdata in zip(subid_list,roi_data[roi]):
        
        preprocedX = deepcopy(subdata["preprocX"])
        stimdf = subdata["stimdf"].copy()
        
        navi_filter = stimdf.taskname.to_numpy() == "navigation"

        #do not average across sessions, so we get a 100*nvoxel matrix
        navi_X = preprocedX[navi_filter,:]
        
        #filter out the df for just one session
        sub_dfall = stimdf[navi_filter].copy().reset_index(drop=True)
        sub_dfall['training_axsetstr'] = ["x" if axs==0 else "y" for axs in sub_dfall['training_axset']]
        # make loc into integer
        sub_dfall["stim_x"] = sub_dfall["stim_x"]*2
        sub_dfall["stim_y"] = sub_dfall["stim_y"]*2
        sub_dfall["stim_xdist"] = sub_dfall["stim_xdist"]*2
        sub_dfall["stim_ydist"] = sub_dfall["stim_ydist"]*2
        sub_dfall["training_axlocTL"] = sub_dfall["training_axlocTL"]*2
        sub_dfall["training_axlocTR"] = sub_dfall["training_axlocTR"]*2

        #generate filters for various stimuli type
        curr_filt = generate_filters(sub_dfall)["training_nocenter"]
        curr_df, curr_X = sub_dfall[curr_filt].copy().reset_index(drop=True), navi_X[curr_filt,:]

        sub_patterns[roi].append(curr_X)
        sub_stimdfs[roi].append(curr_df)


def cal_PS(X,sdf):
    axlocs = np.unique(sdf.training_axlocTL)
    assert all([np.sum(sdf.stim_x == j)==1 for j in axlocs])
    assert all([np.sum(sdf.stim_y == j)==1 for j in axlocs])
    # pick x/y stims and put them into same order
    xstims = np.array([X[sdf.stim_x == j,:][0] for j in axlocs])
    ystims = np.array([X[sdf.stim_y == j,:][0] for j in axlocs])
    return upper_tri(parallel_axes_cosine_sim(xstims,ystims))[0].mean()

def PTA_RSA_CV(subpsX, subrsaX, subdf):
    res_list = []
    PS_fits, PS_evals  = [], []
    runs = subdf.stim_session.to_numpy()
    unique_runs = np.unique(runs)
    for eval_run in unique_runs:
        fit_index = runs!=eval_run
        eval_index = runs==eval_run
        fitX_splits = split_data(subpsX[fit_index,:],groups=runs[fit_index])
        fitPSX  = np.mean(fitX_splits,axis=0)
        
        #subfitPS = np.mean([cal_PS(sX,subdf[subdf.stim_session==0].copy()) for sX in fitX_splits])
        subfitPS = cal_PS(fitPSX,subdf[subdf.stim_session==0].copy())
        
        ## specify analysis
        if subfitPS>0:
            PTAhigh_prim = "PTA_locNomial_TL" 
            PTAlow_prim = "PTA_locEuc_TL"
        elif subfitPS<0:
            PTAhigh_prim = "PTA_locNomial_TR"
            PTAlow_prim = "PTA_locEuc_TR"
        else:
            randidx = np.random.permutation([0,1])
            PTAhigh_prim = ["PTA_locNomial_TL","PTA_locNomial_TR"][randidx]
            PTAlow_prim = ["PTA_locEuc_TL","PTA_locEuc_TR"][randidx]

        analyses = [
            {"name": "train stim Compete PTAhigh Cartesian",
            "reg_names": ["PTA_ax", f"{PTAhigh_prim}", "gtlocEuclidean"],
            "short_names":  ["PTA_ax","PTA_locNomial","gtlocEuclidean"]
            },
            
            {"name": "train stim betweenxy Cartesian", 
            "reg_names": ["betweenxy_gtlocEuclidean"],
            "short_names": ["gtlocEuclidean"]
            },
            {"name": "train stim betweenxy PTA low",
            "reg_names": [f"betweenxy_{PTAlow_prim}"],
            "short_names": ["PTA_locEuc"]
            },
            {"name": "train stim betweenxy PTA high",
            "reg_names": [f"betweenxy_{PTAhigh_prim}"],
            "short_names": ["PTA_locNomial"]
            },
            {"name": "train stim betweenxy Compete PTAhigh Cartesian",
            "reg_names": ["betweenxy_gtlocEuclidean",f"betweenxy_{PTAhigh_prim}"],
            "short_names": ["gtlocEuclidean","PTA_locNomial"],
            },
            {"name": "train stim betweenxy Compete PTAlow Cartesian",
            "reg_names": ["betweenxy_gtlocEuclidean",f"betweenxy_{PTAlow_prim}"],
            "short_names": ["gtlocEuclidean","PTA_locEuc"],
            },
            
        ]
        stim_dict = subdf[subdf.stim_session==0].copy().reset_index(drop=True).to_dict('list')
        submodelrdm = ModelRDM(
            stimid    = stim_dict["stim_id"],
            stimgtloc = np.vstack([stim_dict["stim_x"],stim_dict["stim_y"]]).T,
            stimfeature = np.vstack([stim_dict["stim_color"],stim_dict["stim_shape"]]).T,
            stimgroup = stim_dict["stim_group"],
            sessions = stim_dict["stim_session"],
            nan_identity = False,
            splitgroup  = True
        )
        
        evalRSA_X = subrsaX[eval_index,:] # 
        for analysis_dict in analyses:
            reg_names, short_names = analysis_dict["reg_names"], analysis_dict["short_names"]
            reg_vals  = [submodelrdm.models[k] for k in reg_names]
            reg_estimator  = MultipleRDMRegression(evalRSA_X,
                                                modelrdms=reg_vals,
                                                modelnames=short_names,
                                                rdm_metric="correlation")
            reg_estimator.fit()
            res_df = pd.DataFrame(
                {"modelrdm":   reg_estimator.modelnames,
                "coefficient": reg_estimator.result}
            ).assign(
            analysis = analysis_dict["name"],
            adjR2 = reg_estimator.adjR2,
            R2    = reg_estimator.R2,
            heldoutrun = eval_run
            )
            res_list.append(res_df)
        
        # do PS for heldout as well to see if they aligned
        subevalPS = cal_PS(evalRSA_X,subdf[subdf.stim_session==0].copy())

        PS_fits.append(subfitPS) 
        PS_evals.append(subevalPS) 

    return pd.concat(res_list,axis=0).reset_index(drop=True), PS_fits, PS_evals



#with Parallel(n_jobs=10) as parallel:
ROIPS_PTA_cvres  = {}
ROIRSA_PTA_cvres = []
for pgroi in rois:
    print(f"compute PS according to {pgroi}\n")
    ROIPS_PTA_cvres[pgroi] = dict(zip(rois, [[] for _ in rois]))
    for jsub, (subpsX,subdf,subid) in enumerate(zip(sub_patterns[pgroi],sub_stimdfs[pgroi],subid_list)):
        print(f"RSA in {subid}",end="\r",flush=True)
        #subpsXdemean = np.vstack([scale_feature(x,0) for x in split_data(subpsX,subdf.stim_session)])
        subpsX_oddeven = average_odd_even_session(subpsX,subdf.stim_session)
        subgroup = "Generalizer" if subid in generalizers else "nonGeneralizer"
        subcohort = "first cohort" if subid in cohort1ids else "second cohort"
        
        #for roi in rois:
        roi = pgroi
        subRSAX = deepcopy(subpsX)
        #subrsaXdemean = np.vstack([scale_feature(x,0) for x in split_data(sub_patterns[roi][jsub],subdf.stim_session)])
        subrsaX_oddeven = average_odd_even_session(subRSAX,subdf.stim_session)
        subcvres, subfitPS, subevalPS = PTA_RSA_CV(subpsX_oddeven, subrsaX_oddeven, subdf[subdf.stim_session<2].copy()) #[subdf.stim_session<2].copy()
        #subcvres, subfitPS, subevalPS = PTA_RSA_CV(subpsX, subRSAX, subdf.copy())
        subcvres = subcvres.assign(
            subid=subid,
            subgroup=subgroup,
            cohort = subcohort,
            PSGROUProi  = pgroi,
            RSAroi = roi                
        )
        ROIRSA_PTA_cvres.append(subcvres)
        ROIPS_PTA_cvres[pgroi][roi].append({"fit":subfitPS,"eval":subevalPS})


            

dump({"RSA":ROIRSA_PTA_cvres,"PS":ROIPS_PTA_cvres},os.path.join(ROIRSAdir,"ROI_PTARSA_2v2raw.pkl"))        

# savedop = load(os.path.join(ROIRSAdir,"ROI_PTARSA_cv.pkl"))
# ROIRSA_PTA_cvres = savedop["RSA"]
# ROIPS_PTA_cvres  = savedop["PS"]