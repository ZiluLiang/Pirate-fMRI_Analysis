import pickle
import deepmreye
import numpy as np
import os
from zpyhelper.filesys import checkdir
import json
import sys

project_path = '/home/liangz/pirate_fmri/Analysis'
sys.path.append(project_path)
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Define paths: note that we are running this on wsl, so the scripts we have copied over to wsl but the data is still on the windows side
dminput_dir_win = "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/raw"
dmopt_dir_win   = "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/processed"

# Load experiment metatdata: participant list etc.
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

# Preprocess and save data for deepmreye model evaluation
raw_img_type = "onlyrealigned" # we can try other preprocessing types later, but this is the minimal one and the one used in the paper, have checked the extracted eye mask visually looked ok
n_labels_per_TR = 10 # this is the same as used in the paper, should not matter much here? need to check though
for taskname in ["piratenavigation","localizer"]:
    run_list = [1,2,3,4] if taskname=="piratenavigation" else [1]
    for subid in subid_list:
        subnum = subid.replace("sub","")
        for run in run_list:
            pmask_pat = 'mask_sub-%s_task-%s_run-%d.p'
            subdata_path = os.path.join(dminput_dir_win,raw_img_type,pmask_pat % (subnum, taskname, run))
            subdata = pickle.load(open(subdata_path, "rb"))
            subdata_norm = [deepmreye.preprocess.normalize_img(subdata)]
            labels = [np.zeros((subdata.shape[3],n_labels_per_TR,2))] # we don't actually have the ground truth labels here, just put in dummy zeros
            participant_ids = [([subid]*subdata.shape[3], ['%s_%d'%(taskname, run)]*subdata.shape[3])]
            checkdir(os.path.join(dmopt_dir_win,raw_img_type,subid))
            save_dir = os.path.join(dmopt_dir_win,raw_img_type)
            participant = 'sub-%s_task-%s_run-%d'%(subnum, taskname, run)
            deepmreye.preprocess.save_data(
                                participant,
                                subdata_norm,
                                labels,
                                participant_ids,
                                save_dir,
                                center_labels=False,
                            )