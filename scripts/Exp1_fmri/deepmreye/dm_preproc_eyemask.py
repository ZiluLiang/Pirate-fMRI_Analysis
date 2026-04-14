##################################################################
# repeat the preprocessing step to see which one looks better
##################################################################

import os
from pdb import run
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import deepmreye
import json
from deepmreye.preprocess import run_participant, get_masks
from zpyhelper.filesys import checkdir


project_path = '/home/liangz/pirate_fmri/Analysis'
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")


dm_dir_win   = "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/raw" # extracted eye data and QC plots are also saved here after running the script
raw_img_types = ["onlyrealigned","normunsmoothed","normsmoothed"]
dest_pat = "sub-%s_task-%s_run-%d.nii.gz"

eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges, = get_masks()

for taskname in ["piratenavigation","localizer"]:
    run_list = [1,2,3,4] if taskname=="piratenavigation" else [1]
    for subid in subid_list:
        subnum = subid.replace("sub","")
        for run in run_list:
            for raw_img_type in raw_img_types:
                print(f"\n Processing {taskname} {subid} run{run} with {raw_img_type}...")
                fp_func = os.path.join(dm_dir_win,raw_img_type,dest_pat % (subnum, taskname, run))

                run_participant(
                    fp_func,
                    dme_template,
                    eyemask_big,
                    eyemask_small,
                    x_edges,
                    y_edges,
                    z_edges,
                    transforms=['Affine','Affine','SyNAggro'],
                )
