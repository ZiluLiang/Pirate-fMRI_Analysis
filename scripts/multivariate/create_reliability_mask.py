"""
This script create a reliability mask for each participant based on the method similar to 
Tarhan, L., & Konkle, T. (2020). Reliability-based voxel selection. NeuroImage, 207, 116350.
Before running this script, we built first level glm the include seperate regressors for seperate sessions, yielding 100 regressors.
contrasts were built for any given stimuli in odd and in even runs separately, yielding 50 contrasts.
Here in this script, we extracted the contrast weights as the activity pattern of whole brain voxels in odd and even runs.
This creates two 25*nvoxel matrix, one for odd and one for even. For each voxel, we then correlate the activity pattern vectors(1,25) 
from odd and even runs. The correlation coefficient was used to build a reliability_map.nii saved in the first level directory.
We then use a threshold of r>0 to binarize it into a relaibility mask and save to first level directory.
"""	
import os
import numpy as np
import joblib
from joblib import Parallel, delayed
import nibabel as nib
import nibabel.processing
import matlab.engine
import scipy
from dataloader import ActivityPatternDataLoader
import pandas as pd

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'

## get matlab ready
eng = matlab.engine.start_matlab()

# read in files and set paths
subid_list = eng.eval("get_pirate_defaults(false,'participants').validids")
eng.quit()

fmri_output_path = os.path.join(project_path,'data','fmri')
stim_list_fn = os.path.join(project_path,'scripts','generic','stimlist.txt')
stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0)
stim_id = np.array(stim_list['stim_id'])

glm_name = 'LSA_stimuli_navigation'
LSA_GLM_dir = os.path.join(fmri_output_path,'smoothed5mmLSA',glm_name)
#smoothed5mmLSA
if os.path.exists(os.path.join(LSA_GLM_dir,'wholebrain_oddeven_navistim.pkl')):
    WB_beta = joblib.load(os.path.join(LSA_GLM_dir,'wholebrain_oddeven_navistim.pkl'))
else:
    ## define functions to be run in parallel to extract betas
    def get_WBbeta(LSA_GLM_dir,subid,stim_id):
        # start matlab engine
        eng = matlab.engine.start_matlab()        

        firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)
        contrast_imgo = []
        contrast_imge = []
        for sid in stim_id:
            # call find_contrast_idx function in matlab to find the index of the corresponding contrasts
            eng.evalc("[~,contrast_imgo,~] = find_contrast_idx(fullfile('%s','SPM.mat'),'stim%02d_odd')" % (firstlvl_dir,sid))
            eng.evalc("[~,contrast_imge,~] = find_contrast_idx(fullfile('%s','SPM.mat'),'stim%02d_even')" % (firstlvl_dir,sid))
            contrast_imgo.append(eng.eval("contrast_imgo")) 
            contrast_imge.append(eng.eval("contrast_imge")) 
        nii_fn    = np.concatenate((contrast_imgo,contrast_imge))
        nii_paths = [os.path.join(firstlvl_dir,fn) for fn in nii_fn]
        conds     = np.concatenate((['stim'+str(s)+'_odd' for s in stim_id],['stim'+str(s)+'_even' for s in stim_id]))
        # quit matlab engine
        eng.quit()

        # retrieve pattern data from the contrast images
        # do not drop na values so that the reliavility value can be transformed back to same nifti image dimension
        subWB_beta = ActivityPatternDataLoader(data_nii_paths=nii_paths, conditions=conds,dropNA_flag=False)
        print(f"{subid} activity pattern matrix shape: {subWB_beta.X.shape}")

        return subWB_beta

    WB_beta = Parallel(n_jobs = 5)(delayed(get_WBbeta)(LSA_GLM_dir,subid,stim_id) for subid in subid_list)
    joblib.dump(WB_beta,os.path.join(LSA_GLM_dir,'wholebrain_oddeven_navistim.pkl'))

# voxel selection based on correlation between odd and even runs
def compute_voxel_reliability(subActivityPattern,splits,LSA_GLM_dir,subid):
    _,valid_voxels = subActivityPattern.drop_na() #this is essentially like applying first level mask generated after spm model estimation
    splitted_X,_ = subActivityPattern.split_data(splits)
    pattern_oddrun,pattern_evenrun = splitted_X
    vox_reliability = np.empty((subActivityPattern.X.shape[1],))
    for j in np.arange(subActivityPattern.X.shape[1]):
        if valid_voxels[j]:
            vox_reliability[j] = scipy.stats.spearmanr(pattern_oddrun[:,j], pattern_evenrun[:,j]).correlation
        else:
            vox_reliability[j] = np.nan
    sub_rmaskdata = np.logical_and(vox_reliability>0, ~np.isnan(vox_reliability))

    # load the first level mask image of this participant to get the image dimension header and affine matrix to write into original space
    firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)
    firstlvl_spm_mask = nib.load(os.path.join(firstlvl_dir,'mask.nii'))
    
    # reshape the voxel reliability array into 3D array and create a nifti image in the same first level folder
    sub_rdata_reshaped = np.reshape(vox_reliability,firstlvl_spm_mask.shape)
    sub_rimg  = nib.Nifti1Image(sub_rdata_reshaped, firstlvl_spm_mask.affine,firstlvl_spm_mask.header)
    nib.save(sub_rimg, os.path.join(firstlvl_dir, 'reliability_map.nii')) 
    
    # reshape the voxel reliability mask array into 3D array and create a nifti image in the same first level folder
    sub_rmaskdata_reshaped = np.reshape(1*sub_rmaskdata,firstlvl_spm_mask.shape).astype(np.int8)
    sub_rmaskimg  = nib.Nifti1Image(sub_rmaskdata_reshaped, firstlvl_spm_mask.affine,firstlvl_spm_mask.header)
    nib.save(sub_rmaskimg, os.path.join(firstlvl_dir, 'reliability_mask.nii')) 
    
    vox_retention_rate = np.sum(sub_rmaskdata)/np.sum(valid_voxels)
    print(f"{subid} whole brain voxel retention rate: {vox_retention_rate}")
    
    return {"reliability":vox_reliability,"retentionrate":vox_retention_rate}

splits = np.concatenate((np.ones_like(stim_id),np.ones_like(stim_id)*2))    
WB_reliability = Parallel(n_jobs = 10)(delayed(compute_voxel_reliability)(sAP,splits,LSA_GLM_dir,subid) for (sAP,subid) in zip(WB_beta,subid_list))
joblib.dump(WB_reliability,os.path.join(LSA_GLM_dir,'wholebrain_reliability.pkl'))

import seaborn as sns
import pandas as pd
M = np.array([t["reliability"] for t in WB_reliability])
df = pd.DataFrame({"r":M.ravel(),"sub":np.repeat(np.arange(M.shape[0]), M.shape[1])})
df["row"] = np.floor(df["sub"]/10)
df["col"] = np.mod(df["sub"],10)
r_displot = sns.displot(data = df, x = "r",col="col",row="row")
for ax, sub in zip(r_displot.axes.flat, range(30)):
    txt = "sub%03d" % (sub+1)
    ax.set_title(txt)
r_displot.savefig(os.path.join(LSA_GLM_dir,'WB_reliability_distribution.png'))
