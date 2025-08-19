"""
This script runs RSA analysis in whole-brain searchlight for the high-dimensional REPRESENTATION

"""

import json
import os
import sys
from joblib import cpu_count

from multivariate.mvpa_runner import MVPARunner


###################################################### Run different RSA Analysis  ##################################################
project_path = r'E:\pirate_fmri\Analysis'
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    generalizers = pirate_defaults['participants']['generalizerids']
    nongeneralizers = pirate_defaults['participants']['nongeneralizerids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

analyses_list = [
    {"type":"composition",
     "name":"compositionRetrievalCV_raw",
     "task":"both"
    }
    ]


fmridata_preprocess = "unsmoothedLSA"
cross_task_beta_preproc     = {"MVNN": [None]*2}
 
config_modelrdm_  = {"nan_identity":False, "splitgroup":True}
config_neuralrdm_ = {"preproc":cross_task_beta_preproc, "distance_metric":"correlation"}

beta_dir = {
        "navigation": [os.path.join(fmridata_dir,fmridata_preprocess,'LSA_stimuli_navigation')],
        "localizer":[os.path.join(fmridata_dir,fmridata_preprocess,'LSA_stimuli_localizer')]
        }
beta_dir["both"]   = beta_dir["navigation"] + beta_dir["localizer"]


beta_fname = {
    "navigation":['stimuli_4r.nii'],
    "localizer":['stimuli_1r.nii']
    }
beta_fname["both"] = beta_fname["navigation"] + beta_fname["localizer"]

n_sess={
    "navigation":4,
    "localizer":1
}
n_sess["both"] = n_sess["navigation"]+n_sess["localizer"]

res_dir={}
res_dir["navigation"] = beta_dir["navigation"]*n_sess["navigation"] 
res_dir["localizer"] = beta_dir["localizer"]*n_sess["localizer"]
res_dir["both"] = beta_dir["navigation"]*n_sess["navigation"] + beta_dir["localizer"]*n_sess["localizer"]

res_fname = {}
res_fname["navigation"]  = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["navigation"])]
res_fname["localizer"]  = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["localizer"])]
res_fname["both"] = res_fname["navigation"] + res_fname["localizer"]


ds = beta_dir["both"]
vsmask_dir = ds
vsmask_fname = ['mask.nii']*len(ds)

ds_name = taskname = "both"

RSA = MVPARunner(participants=nongeneralizers,#subid_list,#[:1],
                fmribeh_dir=fmribeh_dir,
                beta_dir=ds, beta_fname=beta_fname[ds_name],
                vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                pmask_dir=ds, pmask_fname=['mask.nii']*len(ds),
                res_dir=res_dir[ds_name], res_fname=res_fname[ds_name],
                anatmasks=[], # to debug run in small ROI: 
                #anatmasks=[os.path.join(os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1andFUNCcluster'),'V1_bilateral.nii')],
                taskname=taskname,
                config_modelrdm = config_modelrdm_,
                config_neuralrdm = config_neuralrdm_)

RSA.run_SearchLight(radius = 10,
                    outputdir = os.path.join(fmridata_dir,fmridata_preprocess,'rsa_searchlight',f'{ds_name}_mvnn_allruns'),
                    analyses = analyses_list,
                    njobs = cpu_count()-2)
