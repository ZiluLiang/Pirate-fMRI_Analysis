"""
This script create a reliability mask for each participant based on the method similar to 
Tarhan, L., & Konkle, T. (2020). Reliability-based voxel selection. NeuroImage, 207, 116350.
Before running this script, we built first level glm the include seperate regressors for seperate sessions, yielding 100 regressors.
contrasts were built for any given stimuli in odd and in even runs separately, yielding 50 contrasts.
Here in this script, we extracted the contrast weights as the activity pattern of whole brain voxels in odd and even runs.
This creates two 25*nvoxel matrix, one for odd and one for even. For each voxel, we then correlate the activity pattern vectors(1,25) 
from odd and even runs. The correlation coefficient was used to build a reliability_map.nii saved in the first level directory.
We then use a threshold of r>0 to binarize it into a reliability mask and save to first level directory.
Meanwhile, a permuted version of reliability mask is created. This mask includes the same number of voxels as the reliability mask, but the masking location is randomly permuted from the reliability mask.

Zilu Liang @HIPlab Oxford
2023
"""	
import os
import sys
import numpy as np
from joblib import Parallel, delayed, dump
import nibabel as nib
import nibabel.processing
from scipy.stats import spearmanr
import json
import pandas as pd
import seaborn as sns

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import checkdir

fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = {'odd':'LSA_stimuli_navigation_concatodd', # 'LSA_stimuli_navigation_concatodd' #'LSA_stimuli_navigation'
            'even':'LSA_stimuli_navigation_concateven'} # 'LSA_stimuli_navigation_concateven' #'LSA_stimuli_navigation'
beta_img = {'odd':'stimuli_odd.nii',
            'even':'stimuli_even.nii'}
reliability_dirname = 'reliability_concat' # 'reliability_concat'  # 'reliability_noconcat'

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    allstimid = pirate_defaults['exp']['allstim']
    n_stim = len(allstimid)

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]
for p in preprocess: 
    print(p)
    output_dir_par = os.path.join(fmri_output_path,p,reliability_dirname)
    checkdir(output_dir_par)
    def getsubWB(subid):
        # retrieve pattern data from the contrast images
        # do not drop na values so that the reliability value can be transformed back to same nifti image dimension
        beta_oddruns  = os.path.join(fmri_output_path,p,glm_name['odd'],'first',subid,beta_img['odd'])
        beta_evenruns = os.path.join(fmri_output_path,p,glm_name['even'],'first',subid,beta_img['even'])
        mask_oddruns  = os.path.join(fmri_output_path,p,glm_name['odd'],'first',subid,'mask.nii')
        mask_evenruns = os.path.join(fmri_output_path,p,glm_name['even'],'first',subid,'mask.nii')

        subWB_beta = ActivityPatternDataLoader(
            [beta_oddruns,beta_evenruns],
            [mask_oddruns,mask_evenruns])
        print(f"{subid} activity pattern matrix shape: {subWB_beta.X.shape}")
        return subWB_beta
    
    WB_beta = Parallel(n_jobs = 10)(
        delayed(getsubWB)(subid) for (subid) in subid_list
        )
    
    # voxel selection based on correlation between odd and even runs
    def compute_voxel_reliability(subAP,outputdir,subid):
        checkdir(outputdir)
        groups = np.concatenate((np.ones((n_stim,)),np.ones((n_stim,))*2)) # split into odd and even
        [pattern_oddrun,pattern_evenrun] = subAP.split_data(groups)
        vox_reliability = [spearmanr(pattern_oddrun[:,j], pattern_evenrun[:,j]).correlation for j in np.arange(subAP.X.shape[1])]
        vox_reliability = np.array(vox_reliability)
        vox_rmaskdata = np.logical_and(vox_reliability>0, ~np.isnan(vox_reliability))
        vox_permrmaskdata = np.random.permutation(vox_rmaskdata)

        _, voxr_img = subAP.create_img(vox_reliability,ensure_finite=True)
        _, voxr_maskimg = subAP.create_img(vox_rmaskdata,ensure_finite=True)
        _, voxr_permmaskimg = subAP.create_img(vox_permrmaskdata,ensure_finite=True)
        nib.save(voxr_img, os.path.join(outputdir, 'reliability_map.nii'))        
        nib.save(voxr_maskimg, os.path.join(outputdir, 'reliability_mask.nii'))
        nib.save(voxr_permmaskimg, os.path.join(outputdir, 'permuted_reliability_mask.nii'))

        vox_retention_rate = np.sum(vox_rmaskdata)/subAP.X.shape[1]
        print(f"{p} - {subid} whole brain voxel retention rate: {vox_retention_rate}")        
        return {"reliability":vox_reliability,"retentionrate":vox_retention_rate}
    
    firstlvl_dirs = [os.path.join(output_dir_par,'first',subid) for subid in subid_list]
    WB_reliability = Parallel(n_jobs = 10)(delayed(compute_voxel_reliability)(sAP,opdir,subid) for (sAP,opdir,subid) in zip(WB_beta,firstlvl_dirs,subid_list))
    dump(WB_reliability,os.path.join(output_dir_par,'wholebrain_reliability.pkl'))

    # create a distribution plot for each participant
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
    r_displot.savefig(os.path.join(output_dir_par,'WB_reliability_distribution.png'))
