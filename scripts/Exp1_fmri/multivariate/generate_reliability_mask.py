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

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import compute_rdm,lower_tri,upper_tri
from zpyhelper.MVPA.preprocessors import scale_feature, average_odd_even_session,normalise_multivariate_noise, split_data, concat_data
from zpyhelper.image.niidatahandler import write_data_to_image, retrieve_data_from_image

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'src'))

fmri_output_path = os.path.join(project_path,'data','Exp1_fmri','fmri')
glm_name = {'odd':'LSA_stimuli_navigation', 
            'even':'LSA_stimuli_navigation'} 
beta_img = {'odd':'stimuli_odd.nii',
            'even':'stimuli_even.nii'}
reliability_dirname = 'reliability_noconcat'

with open(os.path.join(project_path,'scripts','Exp1_fmri','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    allstimid = pirate_defaults['exp']['allstim']
    n_stim = len(allstimid)

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]
for p in preprocess: 
    print(p)
    output_dir_par = os.path.join(fmri_output_path,p,reliability_dirname)
    checkdir(output_dir_par)
    def compute_voxel_reliability(subid):
        # retrieve pattern data from the contrast images
        # do not drop na values so that the reliability value can be transformed back to same nifti image dimension
        beta_oddruns  = os.path.join(fmri_output_path,p,glm_name['odd'],'first',subid,beta_img['odd'])
        beta_evenruns = os.path.join(fmri_output_path,p,glm_name['even'],'first',subid,beta_img['even'])
        mask_oddruns  = os.path.join(fmri_output_path,p,glm_name['odd'],'first',subid,'mask.nii')
        mask_evenruns = os.path.join(fmri_output_path,p,glm_name['even'],'first',subid,'mask.nii')

        subWB_beta = retrieve_data_from_image(
            [beta_oddruns,beta_evenruns],
            [mask_oddruns,mask_evenruns])
        print(f"{subid} whole-brain activity pattern matrix shape: {subWB_beta.shape}")

        # split into odd and even 
        groups = np.concatenate((np.ones((n_stim,)),np.ones((n_stim,))*2)) 
        [pattern_oddrun,pattern_evenrun] = split_data(subWB_beta,groups)

        # calculate voxel reliability
        vox_reliability = np.array([spearmanr(pattern_oddrun[:,j], pattern_evenrun[:,j]).correlation for j in np.arange(subWB_beta.shape[1])])

        # threshold into mask
        vox_rmaskdata = np.logical_and(vox_reliability>0, ~np.isnan(vox_reliability))

        # generate a permuted mask with the same number of masked voxel but at random place
        vox_permrmaskdata = np.random.permutation(vox_rmaskdata)

        # save reliability image, reliability mask image, and permuted reliability mask image
        outputdir = os.path.join(output_dir_par,'first',subid)
        checkdir(outputdir)
        write_data_to_image(data=vox_reliability,
                            mask_imgs=[mask_oddruns,mask_evenruns],
                            ensure_finite=True,outputpath=os.path.join(outputdir, 'reliability_map.nii'))
        
        write_data_to_image(data=vox_rmaskdata,
                            mask_imgs=[mask_oddruns,mask_evenruns],
                            ensure_finite=True,outputpath=os.path.join(outputdir, 'reliability_mask.nii'))

        write_data_to_image(data=vox_permrmaskdata,
                            mask_imgs=[mask_oddruns,mask_evenruns],
                            ensure_finite=True,outputpath=os.path.join(outputdir, 'permuted_reliability_mask.nii'))
        
        # compute voxel retention rate
        vox_retention_rate = np.sum(vox_rmaskdata)/subWB_beta.shape[1]
        print(f"{p} - {subid} whole brain voxel retention rate: {vox_retention_rate}")  
        return {"reliability":vox_reliability,"retentionrate":vox_retention_rate}
    
    
    WB_reliability = Parallel(n_jobs = 10)(delayed(compute_voxel_reliability)(subid) for (subid) in subid_list)
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
