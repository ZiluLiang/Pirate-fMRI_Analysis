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
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois = ["V1_bilateral","PPC_bilateral","vmPFC_bilateral","HPC_bilateral"]

#run RSA analysis
res_df_list = []
taskrdm = {"navigation": {},
           "localizer": {}}
taskdf = {"navigation": {},
          "localizer": {}}
for roi in rois:
    taskrdm["navigation"][roi],taskrdm["localizer"][roi] = [], []
    taskdf["navigation"][roi], taskdf["localizer"][roi] = [], []
    for subdata,subid in zip(roi_data[roi],subid_list):
        print(f"{roi} - {subid}",end="\r",flush=True)
        subgroup = "Generalizer" if subid in generalizers else "nonGeneralizer"
        preprocedX = deepcopy(subdata["preprocX"])
        stimdf = subdata["stimdf"]
        
        navi_filter = stimdf.taskname.to_numpy() == "navigation"
        lzer_filter = stimdf.taskname.to_numpy() == "localizer"

        task_filters = {
            "navigation": navi_filter,
            "localizer": lzer_filter
        }

        for taskname, task_filter in task_filters.items():
            final_filter = task_filter
            #average across sessions
            curr_X = np.mean(split_data(X      = preprocedX[final_filter,:],
                                        groups = stimdf[final_filter].copy().stim_session.to_numpy()),
                            axis=0)
            #select stimdf using only one session because we did averaging
            task_df = stimdf[final_filter].copy().reset_index(drop=True)
            minsess = task_df.stim_session.min()
            curr_df = task_df[(task_df.stim_session==minsess)].copy().reset_index(drop=True)
            # reorder so that the rdm looks intepretable
            new_order = curr_df.sort_values(["training_axset","training_axlocTL","stim_x","stim_y"]).index.to_numpy()
            curr_X = curr_X[new_order,:]
            curr_df = curr_df.iloc[new_order,:].reset_index(drop=True)
            #save them
            taskrdm[taskname][roi].append(
                compute_rdm(
                    curr_X,
                    metric="correlation"
                )
            )
            taskdf[taskname][roi].append(curr_df)
            # get model rdms
            stim_dict = curr_df.to_dict('list')
            submodelrdm = ModelRDM(
                stimid    = stim_dict["stim_id"],
                stimgtloc = np.vstack([stim_dict["stim_x"],stim_dict["stim_y"]]).T,
                stimfeature = np.vstack([stim_dict["stim_color"],stim_dict["stim_shape"]]).T,
                stimgroup = stim_dict["stim_group"],
                sessions = stim_dict["stim_session"],
                nan_identity = False,
                splitgroup  = True
            )

            highDreg_name = "feature"
            lowDreg_name  = "gtlocEuclidean"
            prim_reg_names = [lowDreg_name,highDreg_name]        

            basic_analyses = [
                {"name": "all Compete Feature Cartesian",
                "reg_names": [f"{x}" for x in prim_reg_names],
                "short_names":  prim_reg_names
                },
                {"name": "all Cartesian",
                "reg_names": [lowDreg_name],
                "short_names": [lowDreg_name]
                },
                {"name": "all feature",
                "reg_names": [highDreg_name],
                "short_names": [highDreg_name]
                }

                ]

            if taskname=="navigation":
                analyses = basic_analyses + [
                    {"name": "train stim withinxy - Euc",
                    "reg_names": [f"withinxy_trainstimpairs_{lowDreg_name}"],
                    "short_names": [lowDreg_name]
                    },
                    {"name": "test stim withinxy - Euc",
                    "reg_names": [f"withinxy_teststimpairs_{lowDreg_name}"],
                    "short_names": [lowDreg_name]
                    },

                    {"name": "test stim Compete high low D",
                    "reg_names": [f"teststimpairs_{x}" for x in prim_reg_names],
                    "short_names":  prim_reg_names
                    },

                    {"name": "train stim low D",
                    "reg_names": [f"trainstimpairs_{lowDreg_name}"],
                    "short_names": [lowDreg_name]
                    },
                    {"name": "train stim high D",
                    "reg_names": [f"trainstimpairs_{highDreg_name}"],
                    "short_names": [highDreg_name]
                    },
                    {"name": "test stim low D",
                    "reg_names": [f"teststimpairs_{lowDreg_name}"],
                    "short_names": [lowDreg_name]
                    },
                    {"name": "test stim high D",
                    "reg_names": [f"teststimpairs_{highDreg_name}"],
                    "short_names": [highDreg_name]
                    }
                    
                ]
            else:
                analyses = basic_analyses
            

            for analysis_dict in analyses:
                reg_names = [k for k in analysis_dict["reg_names"] if k in submodelrdm.models.keys()]
                short_names = [sn for sn,k in zip(analysis_dict["short_names"],analysis_dict["reg_names"]) if k in submodelrdm.models.keys()]
                reg_vals  = [submodelrdm.models[k] for k in reg_names]
                if len(reg_vals) == 0:
                    print(f"Skipping {roi} - {subid} - {taskname} - {analysis_dict['name']} because no regressor found")
                    continue
                elif len(reg_vals) == 1:
                    reg_estimator = PatternCorrelation(curr_X,
                                                        modelrdms=reg_vals,
                                                        modelnames=short_names,
                                                        rdm_metric="correlation")
                else:
                    reg_estimator  = MultipleRDMRegression(curr_X,
                                                        modelrdms=reg_vals,
                                                        modelnames=short_names,
                                                        rdm_metric="correlation")
                reg_estimator.fit()
                res_df = pd.DataFrame(
                    {"modelrdm":   reg_estimator.modelnames,
                    "coefficient": reg_estimator.result}
                ).assign(
                roi = roi,
                subid=subid,
                subgroup=subgroup,
                taskname = taskname,
                analysis = analysis_dict["name"],
                adjR2 = reg_estimator.adjR2 if len(reg_vals)>1 else np.nan,
                R2    = reg_estimator.R2 if len(reg_vals)>1 else np.nan
                )
                res_df_list.append(res_df)

res_df = pd.concat(res_df_list,axis=0).reset_index(drop=True)
checkdir(os.path.join(ROIRSAdir,"withintaskrsa"))
res_df.to_csv(os.path.join(ROIRSAdir,"withintaskrsa","rsa_withintask.csv"),index=False)
dump({
    "taskrdm": taskrdm,
    "taskdf": taskdf
}, os.path.join(ROIRSAdir,"withintaskrsa","withintask_rdms.pkl")
)