##################################################################
# copy and compress all the corresponding func files to a new folder for deepmreye preprocessing
##################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import deepmreye
import json
from deepmreye.util import data_generator, model_opts, util
from zpyhelper.filesys import checkdir

# Define paths: note that we are running this on wsl, so the scripts we have copied over to wsl but the data is still on the windows side
project_path = '/home/liangz/pirate_fmri/Analysis'
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

# original data are still kept on windows side, 
# we only move the extracted eye data to wsl side later on (see dm_preproc_preprocess.py)
fmri_dir_win = "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/preprocess"
dm_dir_win   = "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/raw" # note that the extracted eye data will also be saved here after running dm_preproc_eyemask.py
checkdir(dm_dir_win)
file_patterns = {
    "onlyrealigned":  "uosub-%s_task-%s_run-%d.nii",
    "normunsmoothed": "wuosub-%s_task-%s_run-%d.nii",
    "normsmoothed":   "swuosub-%s_task-%s_run-%d.nii",
}
dest_pat = "sub-%s_task-%s_run-%d.nii"
for taskname in ["piratenavigation","localizer"]:
    run_list = [1,2,3,4] if taskname=="piratenavigation" else [1]
    for subid in subid_list:
        subnum = subid.replace("sub","")
        for run in run_list:
            print(f"Moving files for {subid} {taskname} run{run}...",end="\r",flush=True)
            for prep_type, func_pat in file_patterns.items():
                fp_func = os.path.join(fmri_dir_win,subid,func_pat % (subnum, taskname, run))
                dest_dir =  os.path.join(dm_dir_win,prep_type)
                checkdir(dest_dir)
                dest_fp = os.path.join(dest_dir,dest_pat % (subnum, taskname, run))
                if not os.path.exists(dest_fp):
                    cmd = f"gzip -c {fp_func} > {dest_fp}.gz"
                    print(cmd)
                    os.system(cmd)
