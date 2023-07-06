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
from multivariate.helper import ModelRDM, compute_rdm, checkdir, scale_feature
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
from multivariate.rsa_runner import RSARunner



###################################################### Run RSA Analysis in different 'preprocessing pipelines' ##################################################

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

ROIRSA_output_path = os.path.join(fmridata_dir,'ROIRSA','reliability_concat_nanidentity')
checkdir(ROIRSA_output_path)

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]
anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'
with open(os.path.join(project_path,'scripts','anatomical_masks.json')) as f:    
    anat_roi = list(json.load(f).keys())
laterality = ["left","right","bilateral"]

n_sess = {
          "localizer":1,
          "fourruns":4,
          "concatall":1
          }

corr_df_list = []

for p in preprocess:
    beta_dir = {
        "localizer":[os.path.join(fmridata_dir,p,'LSA_stimuli_localizer')],
        "fourruns":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')],
        "concatall":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation_concatall')]
        }
    beta_fname = {
        "localizer":['stimuli_1r.nii'],
        "fourruns":['stimuli_4r.nii'],
        "concatall":['stimuli_all.nii']
        }
    vs_dir = {"no_selection":[],
              "reliability_ths0":[os.path.join(fmridata_dir,p,'reliability_concat')],
              "perm_rmask":[os.path.join(fmridata_dir,p,'reliability_concat')]}
    for ds_name,ds in beta_dir.items():
        mds_df_list = []
        rdm_df_list = []
        for vselect,vdir in vs_dir.items():
            vsmask_dir = ds + vdir
            if vselect == "no_selection":
                vsmask_fname = ['mask.nii']*len(ds)
            elif vselect == "perm_rmask":
                vsmask_fname = ['mask.nii']*len(ds) + ['permuted_reliability_mask.nii']
            elif vselect == "reliability_ths0":
                vsmask_fname = ['mask.nii']*len(ds) + ['reliability_mask.nii']
            taskname = "navigation" if ds_name != "localizer" else ds_name

            for roi, lat in itertools.product(anat_roi, laterality):
                print(f"{p} - {ds_name} - {vselect} - {roi} = {lat}")
                anatmasks = [os.path.join(anatmaskdir,f'{roi}_{lat}.nii')]
                RSA = RSARunner(participants=subid_list,
                                fmribeh_dir=fmribeh_dir,
                                beta_dir=ds, beta_fname=beta_fname[ds_name],
                                vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                                pmask_dir=ds, pmask_fname=beta_fname[ds_name],
                                anatmasks=anatmasks,
                                nsession=n_sess[ds_name],
                                taskname=taskname)
                mdsdf = RSA.run_ROIMDS(10)
                corr_df,rdm_df = RSA.run_ROIRSA(10)

                mdsdf = mdsdf.assign(roi = roi, laterality = lat, voxselect = vselect, preprocess=p,ds = ds_name)
                rdm_df = rdm_df.assign(roi = roi, laterality = lat, voxselect = vselect, preprocess=p,ds = ds_name)
                corr_df = corr_df.assign(roi = roi, laterality = lat, voxselect = vselect, preprocess=p,ds = ds_name)                

                mds_df_list.append(mdsdf)
                rdm_df_list.append(rdm_df)
                corr_df_list.append(corr_df)
        pd.concat(mds_df_list,axis=0).to_csv(os.path.join(ROIRSA_output_path,f'roimds_nocentering_{p}_{ds_name}_{vselect}.csv'))
        pd.concat(rdm_df_list,axis=0).to_csv(os.path.join(ROIRSA_output_path,f'roirdm_nocentering_{p}_{ds_name}_{vselect}.csv'))
pd.concat(corr_df_list, axis=0).to_csv(
    os.path.join(ROIRSA_output_path, 'roicorr_nocentering.csv')
)
    