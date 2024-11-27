"""
This script compresses first level residual files into one gz file so that it is easier to read in by searchlight/roi mvpa analysis
"""
import numpy as np
import json
import os
import sys

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts','Exp1_fmri'))
sys.path.append(os.path.join(project_path,'src'))

import nibabel as nib
import nibabel.processing

with open(os.path.join(project_path,'scripts',"Exp1_fmri",'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

taskname="localizer"
if taskname=="navigation":
    glm_dir = os.path.join(fmridata_dir,"unsmoothedLSA",'LSA_stimuli_navigation')
    cutoff_scan = np.array([0,296,296,296,296]).cumsum()
elif taskname=="localizer":
    glm_dir = os.path.join(fmridata_dir,"unsmoothedLSA",'LSA_stimuli_localizer')
    cutoff_scan = np.array([0,326]).cumsum()

for subid in subid_list:
    print(f"compressing redisual nii for {subid}")
    firstlvl_dir = os.path.join(glm_dir,'first',subid)
    for krun in range(len(cutoff_scan)-1):
        s = cutoff_scan[krun]+1
        e = cutoff_scan[krun+1]
        res_img_names = ['Res_%04d.nii' % x for x in np.arange(e-s+1)+s]
        res_img_list = [os.path.join(firstlvl_dir,rn) for rn in res_img_names]
        
        img = nib.funcs.concat_images(res_img_list)
        output_res_fn = os.path.join(firstlvl_dir,f'resid_run{krun+1}.nii.gz')
        nib.save(img,output_res_fn)