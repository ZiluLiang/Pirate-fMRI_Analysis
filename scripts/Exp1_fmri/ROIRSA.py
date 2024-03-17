"""
This script runs RSA analysis in ROI or in wholebrain searchlight

"""


import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import glob
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import compute_rdm, checkdir, scale_feature
from multivariate.modelrdms import ModelRDM
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
from multivariate.rsa_runner import RSARunner



###################################################### Run RSA Analysis in different 'preprocessing pipelines' ##################################################

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nonlearners = pirate_defaults['participants']['nonlearnerids']
    nongeneralizers = pirate_defaults['participants']['nongeneralizerids']
    
ROIRSA_output_path = os.path.join(fmridata_dir,'ROIRSA','funcroi_respcompete_locwrtcetrefeature')
checkdir(ROIRSA_output_path)


maskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\func\wbsearch_reg_respcompete_locwrtcetrefeature_combinexy'
laterality = ["bilateral"]
roi_dict = dict(zip(
    ['localxy - L Cuneus','localxy - L Mid Cingulate',
     'localxy - R Mid Frontal','localxy - R Sup Parietal', 
     'localxy - Sup SuppMotor', 
     'global - Calcarine',
     ],
    ['xydist_LeftCuneus','xydist_RightCingulateMid',
     'xydist_RightFrontalMid','xydist_RightParietalSup',
     'xydist_SuppMotor','xysign_Calcarine'
     ]
     ))

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]

n_sess = {
          "localizer":1,
          "noconcatall":1,
          "oddeven":2,
          "fourruns":4,
          
          "concatall":1,
          "concateven":1,
          "concatodd":1,
          "concatoddeven":2,          
          
          "fourrunsRESP":4,
          "noconcatallRESP":1,
          "concatallRESP":1,

          "fourrunsBEFORE":4,
          "noconcatallBEFORE":1,
          "concatallBEFORE":1
          }    

corr_df_list = []
n_job = 15
for p in preprocess[:1]:
    beta_dir = {
        "fourruns":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')],
#        "noconcatall":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')],
#        "localizer":[os.path.join(fmridata_dir,p,'LSA_stimuli_localizer')],
#        "oddeven":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')]*2,

        }
    beta_fname = {
        "fourruns":['stimuli_4r.nii'],
        "localizer":['stimuli_1r.nii'],
        "noconcatall":['stimuli_all.nii'],
        "oddeven":['stimuli_odd.nii',
                   'stimuli_even.nii'],
        
        "concatall":['stimuli_all.nii'],
        "concateven":['stimuli_even.nii'],
        "concatodd":['stimuli_odd.nii'],
        "concatoddeven":['stimuli_odd.nii',
                         'stimuli_even.nii'],                         
        
        "fourrunsRESP":['stimuli_4r_resp.nii'],
        "noconcatallRESP":['stimuli_all_resp.nii'],
        "concatallRESP":['stimuli_all_resp.nii'],

        "fourrunsBEFORE":['stimuli_4r_before.nii'],
        "noconcatallBEFORE":['stimuli_all_before.nii'],
        "concatallBEFORE":['stimuli_all_before.nii'],
        }
    vs_dir = {
              "no_selection":[],
              #"reliability_ths0":[os.path.join(fmridata_dir,p,'reliability_noconcat')],
              #"perm_rmask":[os.path.join(fmridata_dir,p,'reliability_noconcat')],
              }
    for ds_name,ds in list(beta_dir.items()):
        for vselect,vdir in vs_dir.items():
            mds_df_list = []
            rdm_df_list = []
            PS_df_list = []

            vsmask_dir = ds + vdir
            if vselect == "no_selection":
                vsmask_fname = ['mask.nii']*len(ds)
            elif vselect == "perm_rmask":
                vsmask_fname = ['mask.nii']*len(ds) + ['permuted_reliability_mask.nii']
            elif vselect == "reliability_ths0":
                vsmask_fname = ['mask.nii']*len(ds) + ['reliability_mask.nii']
            taskname = "navigation" if ds_name != "localizer" else ds_name

            for (roi,roi_fn), lat in itertools.product(roi_dict.items(), laterality):
                print(f"{p} - {ds_name} - {vselect} - {roi} = {lat}")
                anatmasks = [os.path.join(maskdir,f'{roi_fn}_{lat}.nii')]
                anatmasks = [os.path.join(maskdir,f'{roi_fn}.nii')]
                RSA = RSARunner(participants=subid_list,
                                fmribeh_dir=fmribeh_dir,
                                beta_dir=ds, beta_fname=beta_fname[ds_name],
                                vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                                pmask_dir=ds, pmask_fname=beta_fname[ds_name],
                                anatmasks=anatmasks,
                                nsession=n_sess[ds_name],
                                taskname=taskname)
                
                corr_rdm_names = [
                    "feature2d",
                    "gtlocEuclidean",
                    "global_xysign",
                    "locwrtcentre_localxy",
                    "locwrtlrbu_localxy",
                    "respglobal_xysign",
                    "resplocwrtcentre_localxy"
                 ]
                #corr_rdm_names = corr_rdm_names + [f'between_teststimpairs_{x}' for x in corr_rdm_names] + [f'between_{x}' for x in corr_rdm_names]
                roirsa_config = {'corr_rdm_names':corr_rdm_names[:1]}
                _,rdm_df = RSA.run_ROIRSA(n_job,roirsa_config)
                rdm_df = rdm_df.assign(roi = roi, laterality = lat, voxselect = vselect, preprocess=p,ds = ds_name)
                #corr_df = corr_df.assign(roi = roi, laterality = lat, voxselect = vselect, preprocess=p,ds = ds_name)                
                rdm_df_list.append(rdm_df)
                #corr_df_list.append(corr_df)
                
                if n_sess[ds_name]<2:
                    PS_df = RSA.run_ROIPS(outputdir=os.path.join(ROIRSA_output_path,'individual_cossim'),njobs=n_job)
                    PS_df = PS_df.assign(roi = roi, laterality = lat, voxselect = vselect, preprocess=p,ds = ds_name)
                    PS_df_list.append(PS_df)
            
            pd.concat(rdm_df_list,axis=0).to_csv(os.path.join(ROIRSA_output_path,f'roirdm_nocentering_{p}_{ds_name}_{vselect}.csv'))
            if n_sess[ds_name]<2:
                pd.concat(PS_df_list,axis=0).to_csv(os.path.join(ROIRSA_output_path,f'roiPS_nocentering_{p}_{ds_name}_{vselect}.csv'))

#pd.concat(corr_df_list, axis=0).to_csv(
#    os.path.join(ROIRSA_output_path, 'roicorr_nocentering.csv')
#)