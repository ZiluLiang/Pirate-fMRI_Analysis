import os
import numpy as np
import nibabel as nib
import nibabel.processing

from sklearn.model_selection import LeaveOneGroupOut

from nilearn.image import new_img_like
from nilearn.decoding import SearchLight

import matlab.engine
import pandas as pd
import glob

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'
LSA_GLM_dir = os.path.join(fmri_output_path,'unsmoothedLSA',glm_name)

eng = matlab.engine.start_matlab()
subid_list = eng.eval("get_pirate_defaults(false,'participants').validids")

n_run = 4
beta_flist = []
mask_flist = []
pmask_flist = []
run_stim_labels = []
y_dict = {"id":[],
          "image":[],
          "locx":[],
          "locy":[],
          "color":[],
          "shape":[]} 

for subid in subid_list[0:1]:
    print(f"retrieving data from {subid}")

    # load stimuli list
    stim_list_fn = glob.glob(os.path.join(fmri_output_path,'beh',subid,'sub*_stimlist.txt'))[0]
    stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0)
    # get stimuli id
    stim_id = np.array(stim_list['stim_id'])
    # get stimuli image
    stim_image = np.array([x.replace('.png','') for x in stim_list["stim_img"]])
    # get 2d location
    stim_locx = np.array(stim_list['stim_x'])
    stim_locy = np.array(stim_list['stim_y'])
    # get visual features
    stim_color = np.array([x.replace('.png','').split('_')[0] for x in stim_list["stim_img"]])
    stim_shape = np.array([x.replace('.png','').split('_')[1] for x in stim_list["stim_img"]])
    
    # build list of beta maps
    firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)
    stimcon_fn = []
    for sid in stim_id:
        # call find_regressor_idx function in matlab to find the index of the corresponding regressor
        _, betaimg = eng.find_regressor_idx(eng.fullfile(firstlvl_dir,'SPM.mat'),
                                            'stim%02d' % (sid),nargout = 2)
        stimcon_fn.append([os.path.join(firstlvl_dir,x) for x in betaimg])
    
    y_dict["id"].append(np.tile(stim_id,n_run))
    y_dict["image"].append(np.tile(stim_image,n_run))
    y_dict["locx"].append(np.tile(stim_locx,n_run))
    y_dict["locy"].append(np.tile(stim_locy,n_run))
    y_dict["color"].append(np.tile(stim_color,n_run))
    y_dict["shape"].append(np.tile(stim_shape,n_run))

    beta_flist.append(np.vstack(stimcon_fn).T.flatten())    
    mask_flist.append(os.path.join(firstlvl_dir,'mask.nii'))
    pmask_flist.append(os.path.join(firstlvl_dir,'reliability_mask.nii'))
    

from sklearn.svm import LinearSVC
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut

pipeline = make_pipeline(StandardScaler(), LinearSVC(penalty='l2',loss='squared_hinge',max_iter=1000))

#pattern_scaler = FunctionTransformer(zscore, kw_args={'axis': 1}, 
#                                     validate=True)

#pipeline = make_pipeline(pattern_scaler, LinearSVC(penalty='l2',loss='squared_hinge',max_iter=1000))

clf = pipeline

# set up leave-one-run-out cross-validation
loro = LeaveOneGroupOut()
run_labels = np.repeat(np.arange(n_run)+1,len(stim_id))

j = 0
subid = subid_list[j]
anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks'
visual_mask = os.path.join(anatmaskdir,'AAL_Occipital.nii')

anat_mask = visual_mask
from nilearn.masking import apply_mask,intersect_masks
tmp = nib.processing.resample_from_to(nib.load(anat_mask), nib.load(mask_flist[j]))
tmpimg = new_img_like(mask_flist[j],np.where(tmp.get_fdata() > 0 , 1, 0))
anat_maskimg = intersect_masks([tmpimg,nib.load(mask_flist[j])],threshold=1)
searchlight = SearchLight(mask_img=anat_maskimg,
                          radius=10, estimator=clf, 
                          cv=loro,verbose=100,
                          n_jobs=1)
searchlight.fit(beta_flist[j], y_dict["shape"][j], groups=run_labels)
results = new_img_like(mask_flist[j], searchlight.scores_)

nib.save(results, 'testsearchlight_decodeshape_visual.nii')