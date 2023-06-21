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
from scipy.stats import spearmanr
import json
from dataloader import ActivityPatternDataLoader
import pandas as pd
import seaborn as sns

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]
for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)
    print(p)
    def getsubWB(subid):
        # retrieve pattern data from the contrast images
        # do not drop na values so that the reliavility value can be transformed back to same nifti image dimension
        subWB_beta = ActivityPatternDataLoader(
            os.path.join(LSA_GLM_dir,'first',subid,'stimuli_oe.nii'),
            os.path.join(LSA_GLM_dir,'first',subid,'mask.nii'))
        print(f"{subid} activity pattern matrix shape: {subWB_beta.X.shape}")
        return subWB_beta
    WB_beta = Parallel(n_jobs = 10)(
        delayed(getsubWB)(subid) for (subid) in subid_list
        )
    
    # voxel selection based on correlation between odd and even runs
    def compute_voxel_reliability(subAP,outputdir,subid):
        groups = np.concatenate((np.ones((25,)),np.ones((25,))*2))
        [pattern_oddrun,pattern_evenrun] = subAP.split_data(groups)
        vox_reliability = [spearmanr(pattern_oddrun[:,j], pattern_evenrun[:,j]).correlation for j in np.arange(subAP.X.shape[1])]
        vox_reliability = np.array(vox_reliability)
        vox_rmaskdata = np.logical_and(vox_reliability>0, ~np.isnan(vox_reliability))

        _, voxr_img = subAP.create_img(vox_reliability)
        _, voxr_maskimg = subAP.create_img(vox_rmaskdata)
        nib.save(voxr_img, os.path.join(outputdir, 'reliability_map.nii'))        
        nib.save(voxr_maskimg, os.path.join(outputdir, 'reliability_mask.nii')) 

        vox_retention_rate = np.sum(vox_rmaskdata)/subAP.X.shape[1]
        print(f"{subid} whole brain voxel retention rate: {vox_retention_rate}")        
        return {"reliability":vox_reliability,"retentionrate":vox_retention_rate}
    
    firstlvl_dirs = [os.path.join(LSA_GLM_dir,'first',subid) for subid in subid_list]
    WB_reliability = Parallel(n_jobs = 10)(delayed(compute_voxel_reliability)(sAP,opdir,subid) for (sAP,opdir,subid) in zip(WB_beta,firstlvl_dirs,subid_list))
    joblib.dump(WB_reliability,os.path.join(LSA_GLM_dir,'wholebrain_reliability.pkl'))

    M = [t["reliability"] for t in WB_reliability]
    R = [t["retentionrate"] for t in WB_reliability]
    df = pd.DataFrame({"r":np.concatenate(M),
                       "sub": np.concatenate([np.repeat(s,x.size) for s,x in enumerate(M)])})
    
    df["row"] = np.floor(df["sub"]/10)
    df["col"] = np.mod(df["sub"],10)
    r_displot = sns.displot(data = df, x = "r",col="col",row="row")
    for ax, sub, rsub in zip(r_displot.axes.flat, subid_list, R):
        txt = "%s - approx. %0.2f%% voxels' r> 0" % (sub,rsub*100)
        ax.set_title(txt)
    r_displot.savefig(os.path.join(LSA_GLM_dir,'WB_reliability_distribution.png'))
