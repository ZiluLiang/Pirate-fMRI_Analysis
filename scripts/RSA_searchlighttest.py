import numpy as np
import matplotlib.pyplot as plt
import os
import matlab.engine
import pandas as pd
import nibabel as nib
import seaborn as sns
from joblib import Parallel, delayed
import nilearn
#from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
from multivariate.helper import compute_rdm,ModelRDM, checkdir


import matplotlib.colors
def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'

## get matlab ready
eng = matlab.engine.start_matlab()

# read in files and set paths
subid_list = eng.eval("get_pirate_defaults(false,'participants').validids")

fmri_output_path = os.path.join(project_path,'data','fmri')
stim_list_fn = os.path.join(project_path,'scripts','generic','stimlist.txt')
stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0)
stim_id = np.array(stim_list['stim_id'])
stim_loc = np.array(stim_list[['stim_x','stim_y']])
stim_feature = np.array(stim_list[['stim_attrx','stim_attry']])

model_rdm = ModelRDM(stim_id,stim_loc,stim_feature,n_session=2)
#model_rdm.visualize()

glm_name = 'LSA_stimuli_navigation'
LSA_GLM_dir = os.path.join(fmri_output_path,'unsmoothedLSA',glm_name)

def get_neuralpatternimg(subid):
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
    #conds     = np.concatenate((['stim'+str(s)+'_odd' for s in stim_id],['stim'+str(s)+'_even' for s in stim_id]))
    mask_path = os.path.join(firstlvl_dir,'mask.nii')
    return subid,contrast_paths,mask_path

print('retrieving participants contrast image directory\n')
RSA_Searchlight_specs = [get_neuralpatternimg(subid) for subid in subid_list]
# quit matlab engine
eng.quit()
print('finished retrieving participants contrast image directory\n')

#['session', 'within_loc2d', 'between_loc2d',
# 'within_loc1dx', 'between_loc1dx', 'within_loc1dy', 'between_loc1dy',
# 'within_feature2d', 'between_feature2d', 
# 'within_feature1dx', 'between_feature1dx', 'within_feature1dy', 'between_feature1dy']
regress_models = []
for mn in ['within_loc2d','within_feature2d']:
    regress_models.append(model_rdm.models[mn]) 
for subid,n_paths,m_paths in RSA_Searchlight_specs:
    RSAdir = os.path.join(LSA_GLM_dir,'searchlightrsa','withins_reg')    
    subrsa_dir = os.path.join(RSAdir,subid)
    checkdir(subrsa_dir)
    outputregexp = 'beta%04d.nii'
    subRSA = RSASearchLight(n_paths,m_paths,10,MultipleRDMRegression,subrsa_dir,outputregexp,njobs=5)
    subRSA = subRSA.run(model_rdm.models,True)

for mn in ['between_loc2d','between_feature2d']:
    regress_models.append(model_rdm.models[mn]) 
for subid,n_paths,m_paths in RSA_Searchlight_specs:
    RSAdir = os.path.join(LSA_GLM_dir,'searchlightrsa','betweens_reg')    
    subrsa_dir = os.path.join(RSAdir,subid)
    checkdir(subrsa_dir)
    outputregexp = 'beta%04d.nii'
    subRSA = RSASearchLight(n_paths,m_paths,10,MultipleRDMRegression,subrsa_dir,outputregexp,njobs=5)
    subRSA = subRSA.run(model_rdm.models,True)

model_rdm = ModelRDM(stim_id,stim_loc,stim_feature,n_session=2,cv_sess=False)
for mn in ['session','loc2d','feature2d']:
    regress_models.append(model_rdm.models[mn]) 
for subid,n_paths,m_paths in RSA_Searchlight_specs:
    RSAdir = os.path.join(LSA_GLM_dir,'searchlightrsa','aggregate_reg')    
    subrsa_dir = os.path.join(RSAdir,subid)
    checkdir(subrsa_dir)
    outputregexp = 'beta%04d.nii'
    subRSA = RSASearchLight(n_paths,m_paths,10,MultipleRDMRegression,subrsa_dir,outputregexp,njobs=5)
    subRSA = subRSA.run(model_rdm.models,True)
    