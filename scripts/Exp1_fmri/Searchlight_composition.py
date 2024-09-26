"""
This script runs sanity check for RSA analysis in whole-brain searchlight

"""

import json
import os
import sys
from joblib import cpu_count

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.preprocessors import average_odd_even_session,normalise_multivariate_noise

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'src'))
from multivariate.mvpa_runner import RSARunner


###################################################### Run different RSA Analysis  ##################################################
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']


analyses_list = [
    ############################# test for competition between models  ###########################################
    {"type":"composition",
     "name":"compositionRetrieval",
     "task":"both"
    }
    
            ]


fmridata_preprocess = "unsmoothedLSA"
beta_preproc_steps_withmvnn = {"MVNN": [None]*2,  "ATOE": [None]*2}
cross_task_beta_preproc     = {"MVNN": [None]*2, "ATOE": [None]*2}
 
config_modelrdm_ = {"nan_identity":False, "splitgroup":True}

beta_dir = {
        "localizer":[os.path.join(fmridata_dir,fmridata_preprocess,'LSA_stimuli_localizer')],
        "navigation": [os.path.join(fmridata_dir,fmridata_preprocess,'LSA_stimuli_navigation')],
        }
beta_dir["both"]   = sum(list(beta_dir.values()),[])

beta_fname = {
    "localizer":['stimuli_1r.nii'],
    "navigation":['stimuli_4r.nii'],
    }
beta_fname["both"] = sum(list(beta_fname.values()),[])

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
res_fname["both"] = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["navigation"])] + [f'resid_run{j+1}.nii.gz' for j in range(n_sess["localizer"])]

maskdir = os.path.join(fmridata_dir,'masks','anat_AAL3')
    
for analysis in analyses_list:
    task = analysis["task"]
    if task=="both":
        config_neuralrdm= {"preproc":cross_task_beta_preproc, "distance_metric":"correlation"}
    else:
        config_neuralrdm= {"preproc":beta_preproc_steps_withmvnn, "distance_metric":"correlation"}

    RSA = RSARunner(
                    participants=subid_list[:1], 
                    fmribeh_dir=fmribeh_dir,
                    beta_dir   = beta_dir[task],    beta_fname = beta_fname[task],
                    vsmask_dir = beta_dir[task],  vsmask_fname = ['mask.nii']*len(beta_dir[task]),
                    pmask_dir  = beta_dir[task],  pmask_fname  = ['mask.nii']*len(beta_dir[task]),
                    res_dir    = res_dir[task],     res_fname  = res_fname[task],
    #                anatmasks = [],
                    anatmasks=[os.path.join(maskdir,'HPC_left.nii')],
                    taskname  = task,
                    config_modelrdm  = config_modelrdm_,
                    config_neuralrdm = config_neuralrdm
                    )

    RSA.run_SearchLight(radius = 10,
                        outputdir = os.path.join(fmridata_dir,fmridata_preprocess,'rsa_searchlight',f'compositionRetrieval_noselection_mvnn_atoetest'),
                        analyses = [analysis],
                        njobs = 1)#)cpu_count()-6)

            