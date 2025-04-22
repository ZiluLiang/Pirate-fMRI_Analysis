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
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

analyses_localizer = [
    
    {"type":"correlation",
     "name":"HighLowD",
     "modelrdms":["gtlocEuclidean","gtloc1dx","gtloc1dy",
                 "withinxy_gtlocEuclidean","withinx_gtlocEuclidean","withiny_gtlocEuclidean","betweenxy_gtlocEuclidean",
                 "feature"
                 ]
    },
    {"type":"regression",
     "name":"compete_featurecartesian_combinexy",
     "regressors":["feature","gtlocEuclidean"]
    }                  
            ]

analyses_navigation = [
    # {"type":"regression",
    # "name":"compete_featurecartesian_combinexy_teststim",
    # "regressors":["teststimpairs_feature","teststimpairs_gtlocEuclidean"]},
    
    # {"type":"regression",
    #  "name":"compete_featurecartesian_combinexy_trainingstim",
    #  "regressors":["trainstimpairs_feature","trainstimpairs_gtlocEuclidean"]
    # } ,

    # {"type":"regression",
    # "name":"compete_featurecartesian_combinexy_withsg",
    # "regressors":["feature","gtlocEuclidean","stimuligroup"]},

    # {"type":"correlation",
    # "name":"HighLowD",
    # "modelrdms":["trainstimpairs_gtlocEuclidean","trainstimpairs_gtloc1dx","trainstimpairs_gtloc1dy", 
    #              "trainstimpairs_feature",
    #              "withinxy_trainstimpairs_gtlocEuclidean",

    #             "gtlocEuclidean","gtloc1dx","gtloc1dy","withinxy_gtlocEuclidean","betweenxy_gtlocEuclidean",
    #             "feature","feature_color","feature_shape","stimuligroup",
                 
    #             "teststimpairs_gtlocEuclidean", "teststimpairs_gtloc1dx","teststimpairs_gtloc1dy",
    #             "teststimpairs_feature","teststimpairs_feature_color","teststimpairs_feature_shape",
    #             "withinxy_teststimpairs_gtlocEuclidean"
    #              ]},
    ######################## based on response ###########################
        {"type":"regression",
         "name":"resp_compete_featurecartesian_combinexy_teststim",
         "regressors":["teststimpairs_feature","teststimpairs_resplocEuclidean"]},
        
        {"type":"regression",
         "name":"resp_compete_featurecartesian_combinexy_trainingstim",
         "regressors":["trainstimpairs_feature","trainstimpairs_resplocEuclidean"]
        } ,

        {"type":"regression",
         "name":"resp_compete_featurecartesian_combinexy_withsg",
         "regressors":["feature","resplocEuclidean","stimuligroup"]},
        {"type":"correlation",
         "name":"LowDresp",
         "modelrdms":["trainstimpairs_resplocEuclidean","trainstimpairs_resploc1dx","trainstimpairs_resploc1dy", 
                     "withinxy_trainstimpairs_resplocEuclidean",

                     "resplocEuclidean","resploc1dx","resploc1dy","withinxy_resplocEuclidean","betweenxy_resplocEuclidean",
                     
                     "teststimpairs_resplocEuclidean", "teststimpairs_resploc1dx","teststimpairs_resploc1dy",
                     "withinxy_teststimpairs_resplocEuclidean"
                 ]}
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

run_ds = ["fourruns"] # ["localizer","fourruns"]
for nconfig_name,nconfig in config_neuralrdm.items():
    for ds_name in run_ds:
        ds = beta_dir[ds_name]
        print(f'{fmridata_preprocess} - {ds_name} - {nconfig_name}')
        vsmask_dir = ds
        vsmask_fname = ['mask.nii']*len(ds)
        
        taskname = "navigation" if ds_name != "localizer" else ds_name
        analyses_list = analyses_navigation if taskname == "navigation" else analyses_localizer


        RSA = MVPARunner(participants=subid_list,#[:1],
                        fmribeh_dir=fmribeh_dir,
                        beta_dir=ds, beta_fname=beta_fname[ds_name],
                        vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                        pmask_dir=ds, pmask_fname=['mask.nii']*len(ds),
                        res_dir=ds*n_sess[ds_name], res_fname=[f'resid_run{j+1}.nii.gz' for j in range(n_sess[ds_name])],
                        anatmasks=[], # to debug run in small ROI: 
                        #anatmasks=[os.path.join(os.path.join(fmridata_dir,'masks','anat_AAL3'),'HPC_left.nii')],
                        taskname=taskname,
                        config_modelrdm = config_modelrdm_,
                        config_neuralrdm = nconfig)
        
        RSA.run_SearchLight(radius = 10,
                            outputdir = os.path.join(fmridata_dir,fmridata_preprocess,'rsa_searchlight',f'{ds_name}_{nconfig_name}'),
                            analyses = analyses_list,
                            njobs = cpu_count()-2)




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