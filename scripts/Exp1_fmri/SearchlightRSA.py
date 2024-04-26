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

analyses_localizer = [
                 {"type":"regression",
                 "name":"compete_featurecartesian_combinexy",
                 "regressors":["feature2d","gtlocEuclidean"]},

                 {"type":"regression",
                 "name":"compete_featurecartesian_separatexy",
                 "regressors":["feature2d","gtloc1dx","gtloc1dy"]},

                 {"type":"correlation",
                  "name":"all",
                  "modelrdms":["gtlocEuclidean","feature2d","gtloc1dx","gtloc1dy"]}                  
            ]

analyses_navigation = [
    ############################# test for competition between models  ###########################################
                 {"type":"regression",
                 "name":"compete_featurecartesian_combinexy",
                 "regressors":["feature2d","gtlocEuclidean"]},

                {"type":"regression",
                 "name":"compete_featurecartesian_combinexy_teststim",
                 "regressors":["teststimpairs_feature2d","teststimpairs_gtlocEuclidean"]},

                {"type":"regression",
                 "name":"compete_featurecartesian_separatexy_teststim",
                 "regressors":["teststimpairs_feature2d","teststimpairs_gtloc1dx","teststimpairs_gtloc1dy"]},

                {"type":"regression",
                 "name":"compete_featurecartesian_separatexy_withsg",
                 "regressors":["feature2d","gtloc1dx","gtloc1dy","stimuligroup"]},

                {"type":"regression",
                 "name":"compete_featurecartesian_combinexy_withsg",
                 "regressors":["feature2d","gtlocEuclidean","stimuligroup"]},

                {"type":"regression",
                 "name":"resp-compete_featurecartesian_combinexy_teststim",
                 "regressors":["teststimpairs_feature2d","teststimpairs_resplocEuclidean"]},

                {"type":"regression",
                 "name":"resp-compete_featurecartesian_combinexy_withsg",
                 "regressors":["feature2d","resplocEuclidean","stimuligroup"]},

    ############################# test for hierarchical model  ###########################################

                {"type":"regression",
                 "name":"compete_hierarchical_combinexy",
                 "regressors":["feature2d","stimuligroup","global_xysign","locwrtcentre_localxy","locwrtlrbu_localxy"]},

                {"type":"regression",
                 "name":"compete_hierarchical_combinexy_teststim",
                 "regressors":["teststimpairs_feature2d","teststimpairs_global_xysign","teststimpairs_locwrtcentre_localxy","teststimpairs_locwrtlrbu_localxy"]},

            ]


fmridata_preprocess = "unsmoothedLSA"
beta_preproc_steps_withmvnn= {
                "MVNN": [normalise_multivariate_noise,{}
                        ],
                "AOE": [average_odd_even_session,{}]
                }

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

run_ds = ["fourruns"]#,"localizer"]
for nconfig_name,nconfig in config_neuralrdm.items():
    for ds_name in run_ds:
        ds = beta_dir[ds_name]
        print(f'{fmridata_preprocess} - {ds_name} - {nconfig_name}')
        vsmask_dir = ds
        vsmask_fname = ['mask.nii']*len(ds)
        
        taskname = "navigation" if ds_name != "localizer" else ds_name
        analyses_list = analyses_navigation if taskname == "navigation" else analyses_localizer

        maskdir = os.path.join(fmridata_dir,'masks','anat')
        RSA = RSARunner(participants=subid_list,#,#[:1]
                        fmribeh_dir=fmribeh_dir,
                        beta_dir=ds, beta_fname=beta_fname[ds_name],
                        vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                        pmask_dir=ds, pmask_fname=['mask.nii']*len(ds),
                        res_dir=ds*n_sess[ds_name], res_fname=[f'resid_run{j+1}.nii.gz' for j in range(n_sess[ds_name])],

                        anatmasks=[],
#                        anatmasks=[os.path.join(maskdir,'hippocampus_left.nii')],
                        nsession=n_sess[ds_name],
                        taskname=taskname,
                        config_modelrdm = config_modelrdm_,
                        config_neuralrdm = nconfig)
        
        RSA.run_SearchLightRSA(radius = 10,
                                outputdir = os.path.join(fmridata_dir,fmridata_preprocess,'rsa_searchlight',f'{ds_name}_noselection_{nconfig_name}'),
                                analyses = analyses_list,
                                njobs = cpu_count()-4)
        
        