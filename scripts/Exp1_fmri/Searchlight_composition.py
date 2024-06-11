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
from multivariate.rsa_runner import RSARunner


###################################################### Run different RSA Analysis  ##################################################
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']


analyses_both = [
    ############################# test for competition between models  ###########################################
    {"type":"location_composition",
    "name":"loc2stim"
    }#,
    #{"type":"feature_composition",
    #"name":"train2test"
    #}
    
            ]


fmridata_preprocess = "unsmoothedLSA"
beta_preproc_steps_withmvnn= {"MVNN": [None]*2, "ATOE": [None]*2}
config_neuralrdm= {
    "mvnn_aoe": {"preproc":beta_preproc_steps_withmvnn, "distance_metric":"correlation"},
}
config_modelrdm_ = {"nan_identity":False, "splitgroup":True}

n_sess = {
          "localizer":1,
          "fourruns":4
          }

beta_dir = {
        "localizer":[os.path.join(fmridata_dir,fmridata_preprocess,'LSA_stimuli_localizer')],
        "fourruns": [os.path.join(fmridata_dir,fmridata_preprocess,'LSA_stimuli_navigation')],
        }
beta_fname = {
    "localizer":['stimuli_1r.nii'],
    "fourruns":['stimuli_4r.nii'],
    }

beta_dir = {
    "navigation":[os.path.join(fmridata_dir,'unsmoothedLSA','LSA_stimuli_navigation')],
    "localizer": [os.path.join(fmridata_dir,'unsmoothedLSA','LSA_stimuli_localizer')],
}
beta_fname = {
    "navigation":['stimuli_4r.nii'],
    "localizer":['stimuli_1r.nii']
}
n_sess={
    "navigation":4,
    "localizer":1
}
cross_task_beta_dir   = sum(list(beta_dir.values()),[])
cross_task_beta_fname = sum(list(beta_fname.values()),[])
cross_task_vsmask_dir = cross_task_beta_dir
cross_task_vsmask_fname = ['mask.nii']*len(cross_task_vsmask_dir)
cross_task_res_dir   = beta_dir["navigation"]*n_sess["navigation"] + beta_dir["localizer"]*n_sess["localizer"]
cross_task_res_fname = [f'resid_run{j+1}.nii.gz' for j in range(n_sess["navigation"])] + [f'resid_run{j+1}.nii.gz' for j in range(n_sess["localizer"])]

config_modelrdm_ = {"nan_identity":False, "splitgroup":True}
cross_task_beta_preproc = {"preproc":{"MVNN": [None]*2,
                                      "ATOE": [None]*2}, 
                           "distance_metric":"correlation"}


analyses_list = analyses_both

maskdir = os.path.join(fmridata_dir,'masks','anat')
RSA = RSARunner(
                participants=subid_list,#[:1], 
                fmribeh_dir=fmribeh_dir,
                beta_dir   = cross_task_beta_dir,    beta_fname   = cross_task_beta_fname,
                vsmask_dir = cross_task_vsmask_dir,  vsmask_fname = cross_task_vsmask_fname,
                pmask_dir  = cross_task_vsmask_dir,  pmask_fname  = cross_task_vsmask_fname,
                res_dir    = cross_task_res_dir, 
                res_fname = cross_task_res_fname,
                anatmasks = [],
#                anatmasks=[os.path.join(maskdir,'parahippocampus_left.nii')],
                taskname  = "both",
                config_modelrdm  = config_modelrdm_,
                config_neuralrdm = cross_task_beta_preproc
                )

RSA.run_SearchLight(radius = 10,
                        outputdir = os.path.join(fmridata_dir,fmridata_preprocess,'rsa_searchlight',f'crosstask_noselection_mvnn_atoe'),
                        analyses = analyses_list,
                        njobs = cpu_count()-4)

        