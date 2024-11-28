"""
This script runs RSA analysis in whole-brain searchlight for the Parallel Training Axis (PTA) REPRESENTATION

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
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

analyses_localizer = [
    {"type":"regression",
    "name":"compete_CartesianPTAhighD_trainstim_behavaligned",
    "regressors":["gtlocEuclidean","PTA_ax","PTA_locNomial"]},

    {"type":"regression",
    "name":"compete_CartesianPTAhighD_trainstim_betweenax_behavaligned",
    "regressors":["betweenxy_gtlocEuclidean","betweenxy_PTA_ax","betweenxy_PTA_locNomial"]},

    {"type":"correlation",
    "name":"allwithPTA",
    "modelrdms":["PTA_TL","PTA_TR","PTA_ax","PTA_locEuc_TR","PTA_locEuc_TL","PTA_locNomial_TR","PTA_locNomial_TL",
                 
                 "gtlocEuclidean","gtloc1dx","gtloc1dy",
                 "withinxy_gtlocEuclidean","withinx_gtlocEuclidean","withiny_gtlocEuclidean","betweenxy_gtlocEuclidean",
                 "feature"
                 ]}                  
            ]

analyses_navigation = [
    {"type":"correlation",
    "name":"PTA_behavaligned",
    "modelrdms":["trainstimpairs_gtlocEuclidean","trainstimpairs_gtloc1dx","trainstimpairs_gtloc1dy", 
                 "trainstimpairs_feature",
                 "trainstimpairs_PTA_TL","trainstimpairs_PTA_TR", "trainstimpairs_PTA_ax",
                 "trainstimpairs_PTA_locEuc_TL","trainstimpairs_PTA_locEuc_TL",
                 "trainstimpairs_PTA_locNomial_TR","trainstimpairs_PTA_locNomial_TR",
                 "betweenxy_trainstimpairs_PTA_locEuc_TL","betweenxy_trainstimpairs_PTA_locEuc_TL",
                 "betweenxy_trainstimpairs_PTA_locNomial_TR","betweenxy_trainstimpairs_PTA_locNomial_TR",
                 "withinxy_trainstimpairs_gtlocEuclidean",

                "gtlocEuclidean","gtloc1dx","gtloc1dy","withinxy_gtlocEuclidean","betweenxy_gtlocEuclidean",
                "feature","feature_color","feature_shape","stimuligroup",
                 
                "teststimpairs_gtlocEuclidean", "teststimpairs_gtloc1dx","teststimpairs_gtloc1dy",
                "teststimpairs_feature","teststimpairs_feature_color","teststimpairs_feature_shape",
                "withinxy_teststimpairs_gtlocEuclidean","betweenxy_teststimpairs_gtlocEuclidean",
                 ]},     

    ############################# test for competition between models  ###########################################
   {"type":"regression",
       "name":"compete_featurecartesian_combinexy_teststim",
       "regressors":["teststimpairs_feature","teststimpairs_gtlocEuclidean"]},

   {"type":"regression",
    "name":"compete_featurecartesian_combinexy_withsg",
    "regressors":["feature","gtlocEuclidean","stimuligroup"]},

   {"type":"regression",
   "name":"compete_CartesianPTAhighD_trainstim",
   "regressors":["trainstimpairs_gtlocEuclidean","trainstimpairs_PTA_ax","trainstimpairs_PTA_locNomial"]},

     {"type":"regression",
      "name":"compete_CartesianPTAhighD_trainstim_behavaligned",
      "regressors":["trainstimpairs_gtlocEuclidean","trainstimpairs_PTA_ax","trainstimpairs_PTA_locNomial"]},
     {"type":"regression",
      "name":"compete_CartesianPTAhighD_trainstim_betweenax_behavaligned",
      "regressors":["betweenxy_trainstimpairs_gtlocEuclidean","betweenxy_trainstimpairs_PTA_locNomial"]}
            ]


fmridata_preprocess = "unsmoothedLSA"
beta_preproc_steps_withmvnn= {"MVNN": [None]*2, "ATOE": [None]*2}
config_neuralrdm= {
    "mvnn_averageall": {"preproc":beta_preproc_steps_withmvnn, "distance_metric":"correlation"},
}
config_modelrdm_ = {"nan_identity":True, "splitgroup":True}

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

run_ds = ["localizer"] # ["localizer","fourruns"]
for nconfig_name,nconfig in config_neuralrdm.items():
    for ds_name in run_ds:
        ds = beta_dir[ds_name]
        print(f'{fmridata_preprocess} - {ds_name} - {nconfig_name}')
        vsmask_dir = ds
        vsmask_fname = ['mask.nii']*len(ds)
        
        taskname = "navigation" if ds_name != "localizer" else ds_name
        analyses_list = analyses_navigation if taskname == "navigation" else analyses_localizer

        project_path


        RSA = MVPARunner(participants=subid_list,#[:1],
                        fmribeh_dir=fmribeh_dir,
                        beta_dir=ds, beta_fname=beta_fname[ds_name],
                        vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                        pmask_dir=ds, pmask_fname=['mask.nii']*len(ds),
                        res_dir=ds*n_sess[ds_name], res_fname=[f'resid_run{j+1}.nii.gz' for j in range(n_sess[ds_name])],
                        anatmasks=[], # to debug run in small ROI: 
                        #anatmasks=[os.path.join(os.path.join(fmridata_dir,'masks','marsbarHCP'),'hippocampus_left.nii')],
                        taskname=taskname,
                        config_modelrdm = config_modelrdm_,
                        config_neuralrdm = nconfig)
        
        RSA.run_SearchLight(radius = 10,
                            outputdir = os.path.join(fmridata_dir,fmridata_preprocess,'rsa_searchlight',f'{ds_name}_noselection_{nconfig_name}'),
                            analyses = analyses_list,
                            njobs = cpu_count()-6)




# """ check rdm by running the following code
# import numpy as np
# from scipy.spatial.distance import squareform
# from zpyhelper.MVPA.rdm import lower_tri
# import seaborn as sns
# json_dir = r"E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\unsmoothedLSA\rsa_searchlight\fourruns_noselection_mvnn_averageall\regression\compete_CartesianPTAhighD_trainstim\first\sub001"
# with open(os.path.join(json_dir,"searchlight.json")) as f:
#     subsearchlightconfig = json.load(f)

# lt = subsearchlightconfig["estimator"]["modelRDMs"]
# list(lt.keys())
# checkrdm = lt["trainstimpairs_PTA_locNomial"]
# pmat = np.full_like(squareform(checkrdm),fill_value=np.nan)
# pmat[lower_tri(pmat)[1]] = checkrdm
# sns.heatmap(pmat)
# """