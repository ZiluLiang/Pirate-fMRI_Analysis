import numpy as np
import os
import matlab.engine
import pandas as pd
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import MultipleRDMRegression
from multivariate.helper import ModelRDM

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'

eng = matlab.engine.start_matlab()
subid_list = eng.eval("get_pirate_defaults(false,'participants').validids")
#eng.quit()

fmri_output_path = os.path.join(project_path,'data','fmri')
stim_list_fn = os.path.join(project_path,'scripts','generic','stimlist.txt')
stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0)
stim_id = np.array(stim_list['stim_id'])
stim_loc = np.array(stim_list[['stim_x','stim_y']])
stim_feature = np.array(stim_list[['stim_attrx','stim_attry']])


glm_name = 'LSA_stimuli_navigation'
LSA_GLM_dir = os.path.join(fmri_output_path,'smoothed5mmLSA',glm_name)

def get_oddeven_img(subid):
    #eng = matlab.engine.start_matlab()
    firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)
    contrast_imgo = []
    contrast_imge = []
    for sid in stim_id:
        # call find_contrast_idx function in matlab to find the index of the corresponding contrasts
        eng.evalc("[~,contrast_imgo,~] = find_contrast_idx(fullfile('%s','SPM.mat'),'stim%02d_odd')" % (firstlvl_dir,sid))
        eng.evalc("[~,contrast_imge,~] = find_contrast_idx(fullfile('%s','SPM.mat'),'stim%02d_even')" % (firstlvl_dir,sid))
        contrast_imgo.append(eng.eval("contrast_imgo")) 
        contrast_imge.append(eng.eval("contrast_imge"))
    contrast_fns   = np.concatenate((contrast_imgo,contrast_imge))
    contrast_paths = [os.path.join(firstlvl_dir,fn) for fn in contrast_fns]
    conditions     = np.concatenate((['stim'+str(s)+'_odd' for s in stim_id],['stim'+str(s)+'_even' for s in stim_id]))
    mask_path = os.path.join(firstlvl_dir,'mask.nii')
    #eng.quit()
    return subid,contrast_paths,mask_path,conditions

def get_contrast_img(subid):
    #eng = matlab.engine.start_matlab()
    firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)
    stimcon_fn = []
    for sid in stim_id:
        # call find_regressor_idx function in matlab to find the index of the corresponding regressor
        eng.evalc("[~,contrast_img,~] = find_contrast_idx(fullfile('%s','SPM.mat'),{regexpPattern('^stim%02d$')})" % (firstlvl_dir,sid))
        stimcon_fn.append(eng.eval("contrast_img")) 
    #eng.quit()
    conditions = ['stim%02d' % (s) for s in stim_id]
    mask_path = os.path.join(firstlvl_dir,'mask.nii')
    stimcon_paths = [os.path.join(firstlvl_dir,fn) for fn in stimcon_fn]
    return subid,stimcon_paths,mask_path,conditions

def get_sessionreg_img(subid):
    #eng = matlab.engine.start_matlab()
    firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)
    stimreg_fn = []
    for sid in stim_id:
        # call find_regressor_idx function in matlab to find the index of the corresponding regressor
        eng.evalc("[~,regimgnames] = find_regressor_idx(fullfile('%s','SPM.mat'),'stim%02d')" % (firstlvl_dir,sid))
        stimreg_fn.append(eng.eval("regimgnames")) 
    #eng.quit()
    conditions = ['stim%02d_run%d' % (s,r+1) for s in stim_id for r in range(4)]
    mask_path = os.path.join(firstlvl_dir,'mask.nii')
    stimreg_fn = np.concatenate(stimreg_fn, axis=0)
    stimreg_paths = [os.path.join(firstlvl_dir,fn) for fn in stimreg_fn]
    return subid,stimreg_paths,mask_path,conditions

RSA_specs = dict()
## run on odd and even separated
print('retrieving participants contrast image directory\n')
RSA_specs["oddeven"] = [get_oddeven_img(subid) for subid in subid_list]
print('finished retrieving participants contrast image directory\n')

## run on four separated sessions
print('retrieving participants regressor image directory\n')
RSA_specs["allsess"] = [get_sessionreg_img(subid) for subid in subid_list]
print('finished retrieving participants regressor image directory\n')

eng.quit()

model_rdm =dict()
model_rdm["oddeven"] = ModelRDM(stim_id,stim_loc,stim_feature,n_session=2)
model_rdm["allsess"] = ModelRDM(stim_id,stim_loc,stim_feature,n_session=4)

analysis = {"sc_betweens_stimuli":   ['between_stimuli'],
            "sc_alls_stimuli":       ['stimuli'],
            "sc_withins_feature2d":  ['within_feature2d'],
            "sc_betweens_feature2d": ['between_feature2d'],
            "sc_alls_feature2d":     ['feature2d'],
            "betweens_loc2d":  ["between_loc2d"],
            "withins_loc2d":   ["within_loc2d"],
            "alls_loc2d":      ["loc2d"],
            "betweens_loc1d":  ["between_loc1dx","between_loc1dy"],
            "withins_loc1d":   ["within_loc1dx","within_loc1dy"],
            "alls_loc1d":      ["loc1dx","loc1dy"],
        }
for vselect in ["wholebrain","reliability_ths0"]:
    for k,v in RSA_specs.items():
        for subid,n_paths,m_paths,_ in v:
            print(f'{vselect} - {k}: Running Searchlight RSA in {subid}')
            if vselect == "reliability_ths0":
                m_paths = os.path.join(os.path.dirname(m_paths),'reliability_mask.nii')
            outputregexp = 'beta_%04d.nii'
            subRSA = RSASearchLight(n_paths,m_paths,10,MultipleRDMRegression,njobs=10)

            for a_n,a_reg in analysis.items():
                print(f'Analysis - {a_n}')
                regress_models = []
                for mn in a_reg:
                    regress_models.append(model_rdm[k].models[mn])
                output_dir = os.path.join(LSA_GLM_dir,'searchlightrsa_sc',vselect,k,a_n,'first')
                subRSA.run(regress_models,a_reg,os.path.join(output_dir,subid), outputregexp, True)            
            
            print(f'{k}: Completed searchlight in {subid}')