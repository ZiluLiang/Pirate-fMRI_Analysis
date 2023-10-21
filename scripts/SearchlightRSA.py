"""
This script runs RSA analysis in ROI or in wholebrain searchlight

"""

import json
import os
import sys
from joblib import cpu_count

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.helper import checkdir
from multivariate.rsa_runner import RSARunner

###################################################### Run RSA Analysis in different 'preprocessing pipelines' ##################################################

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]

analyses_localizer = [
                 {"type":"regression",
                 "name":"gtloc_xy",
                 "regressors":["gtloc1dx", "gtloc1dy"]},
                 
                 {"type":"correlation",
                  "name":"all"}#"modelrdms":None                  
            ]
analyses_navigation = [
                {"type":"regression",
                 "name":"loc_wrt_train_4pred",
                 "regressors":["locwrttrain_xdist", "locwrttrain_xsign","locwrttrain_ydist","locwrttrain_ysign"]},
                 
                 {"type":"regression",
                 "name":"loc_wrt_train_2pred",
                 "regressors":["locwrttrain_xydist", "locwrttrain_xysign"]},

                 {"type":"regression",
                 "name":"gtloc_xy",
                 "regressors":["gtloc1dx", "gtloc1dy"]},

                 {"type":"regression",
                 "name":"resploc_xy",
                 "regressors":["resploc1dx", "resploc1dy"]},
                 
                 {"type":"correlation",
                  "name":"all"}
            ]

n_sess = {
          "localizer":1,
          "concatall":1,
          "noconcatall":1,
          "concateven":1,
          "concatodd":1,
          "oddeven":2,
          "concatoddeven":2,
          "fourruns":4
          }
for p in preprocess[:1]:
    corr_df_list = []
    beta_dir = {
    #        "localizer":[os.path.join(fmridata_dir,p,'LSA_stimuli_localizer')],
    #        "concatall":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation_concatall')],
    #        "oddeven":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')]*2,
    #        "noconcatall":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')],
            "fourruns":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')],
            }
    beta_fname = {
        "localizer":['stimuli_1r.nii'],
        "concatall":['stimuli_all.nii'],
        "noconcatall":['stimuli_all.nii'],
        "concateven":['stimuli_even.nii'],
        "concatodd":['stimuli_odd.nii'],
        "fourruns":['stimuli_4r.nii'],
        "oddeven":['stimuli_odd.nii',
                   'stimuli_even.nii'],
        "concatoddeven":['stimuli_odd.nii',
                         'stimuli_even.nii']
        }
    vs_dir = {
#        "no_selection":[],
        "reliability_ths0":[os.path.join(fmridata_dir,p,'reliability_concat')],
        "perm_rmask":[os.path.join(fmridata_dir,p,'reliability_concat')]
        }
    for ds_name,ds in beta_dir.items():
        for vselect,vdir in vs_dir.items():
            print(f'{p} - {ds_name} - {vselect}')
            vsmask_dir = ds + vdir
            if vselect == "no_selection":
                vsmask_fname = ['mask.nii']*len(ds)
            elif vselect == "perm_rmask":
                vsmask_fname = ['mask.nii']*len(ds) + ['permuted_reliability_mask.nii']
            elif vselect == "reliability_ths0":
                vsmask_fname = ['mask.nii']*len(ds) + ['reliability_mask.nii']

            taskname = "navigation" if ds_name != "localizer" else ds_name
            analyses_list = analyses_navigation if taskname == "navigation" else analyses_localizer

            RSA = RSARunner(participants=subid_list,
                            fmribeh_dir=fmribeh_dir,
                            beta_dir=ds, beta_fname=beta_fname[ds_name],
                            vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                            pmask_dir=ds, pmask_fname=['mask.nii']*len(ds),
                            anatmasks=[],
                            nsession=n_sess[ds_name],
                            taskname=taskname)
            RSA.run_SearchLightRSA(radius = 10,
                                   outputdir = os.path.join(fmridata_dir,p,'rsa_searchlight',f'{ds_name}_{vselect}'),
                                   analyses = analyses_list,
                                   njobs = cpu_count()-2)
            