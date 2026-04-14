""" 
This s
"""

import numpy as np
import os
from zpyhelper.filesys import checkdir
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd

# Import DeepMReye
from deepmreye import preprocess, train, analyse,architecture
from deepmreye.util import model_opts, data_generator

from joblib import dump, load, Parallel, delayed

project_path = '/home/liangz/pirate_fmri/Analysis'
sys.path.append(project_path)
import warnings
from dm_utils import infer_gaze_position, compute_corr_with_lags, create_event_timecourse, corr_significance_label,plot_corr_with_lags
warnings.simplefilter('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silent TF GPU warnings

# Load experiment metatdata: participant list etc.
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")


# Define path: note that we are running this on wsl, so the scripts we have copied over to wsl but the data is still on the windows side
raw_img_type = "onlyrealigned"
processed_data_dir = f"/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/processed/{raw_img_type}/"
res_output_dir = f"/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/results/{raw_img_type}/"
checkdir(res_output_dir)
pretrained_weights_dir = os.path.join(project_path,"scripts","Exp1_fmri","deepmreye","pretrained_model_weights_FreyNau")
checkdir(pretrained_weights_dir)   

download_weights = False
if download_weights:
    from osfclient.api import OSF

    # CONFIG
    project_id = "mrhk9"
    remote_folder = "model_weights"  # change this to the folder path you want on OSF
    local_dir = "pretrained_model_weights_FreyNau/"

    # Make sure local path exists
    os.makedirs(local_dir, exist_ok=True)

    # Initialize OSF client
    client = OSF()
    project = client.project(project_id)

    storage = project.storage('osfstorage')  # use OSF Storage provider

    for folder in storage.folders:
        if folder.name == remote_folder:
            for rem_file in folder.files:
                print(f"downloading pretrained weights: {folder.name}/{rem_file.name}")
                with open(os.path.join(local_dir,rem_file.name), "wb") as f:
                    rem_file.write_to(f)

pretrained_models  = dict(zip(
    ['m1_guidedfixations','m2_pursuit','m3_pursuit','m4_pursuit',
     'm5_freeviewing','m6_openclosed','m7_multiDS1to5','m8_multiDS1to6'],
    ['dataset1_guided_fixations.h5',
     'dataset2_pursuit.h5',
     'dataset3_pursuit.h5',
     'dataset4_pursuit.h5',
     'dataset5_free_viewing.h5',
     'dataset6_openclosed.h5',
     'datasets_1to5.h5',
     'datasets_1to6.h5']
))

# inference over all participants
run_inference = False    
if run_inference:
    print("=================================================== Running Inference ===============================================")
    innerTS = [10] # can only use 10 because we are using the pretrained model, otherwise will throw error
    with Parallel(n_jobs=10) as parallel:
        for mname, pretrained_weights_fn in pretrained_models.items():
            print(f"Running Inference using model pretrained on {mname}")
            pretrained_model_fn = f'{pretrained_weights_dir}/{pretrained_weights_fn}'
            checkdir(os.path.join(res_output_dir,mname))
            predictions = {}
            for itvalue in innerTS:
                # loop over task
                predictions[itvalue] = {}

                # first we get the preproc filepath for each sub in each task run
                print("Gathering data")
                task_fns = {}
                for taskname in ["localizer", "piratenavigation"]:
                    run_list = [1] if taskname=="localizer" else [1,2,3,4]
                    n_TR = 326 if taskname == "localizer" else 296
                    task_fns[taskname] = []
                    for run in run_list:
                        subfps = [os.path.join(processed_data_dir,'sub-%s_task-%s_run-%d.npz'%(subid.replace("sub",""), taskname, run)) for subid in subid_list]
                        task_fns[taskname].append(subfps)
                        
                
                # Then we loop over task and participants                
                for taskname in ["localizer", "piratenavigation"]:        
                    predictions[itvalue][taskname] = {}
                    run_list = [1] if taskname=="localizer" else [1,2,3,4]
                    for subid in subid_list:
                        predictions[itvalue][taskname][subid] = [[] for _ in run_list]
                    for run in run_list:
                        print(f"running inference on task {taskname} run {run}")            
                        subfps = task_fns[taskname][run-1]
                        # loop over participant in parallel
                        results = parallel(delayed(infer_gaze_position)(subfp,itvalue,pretrained_model_fn) for subfp in subfps)
                        # distribute it into per participant for saving
                        for subid,res in zip(subid_list,results):
                            predictions[itvalue][taskname][subid][run-1] = res

            dump(predictions, os.path.join(res_output_dir,mname,'DM_predictions.pkl'))        

# plot predictions for each participant
plot_individualtraces = False
if plot_individualtraces:
    print("=================================================== Plotting Inference ===============================================")
    for mname, pretrained_weights_fn in pretrained_models.items():
        print(f"Plotting Inference from model pretrained on {mname}")
    
        predictions = load(os.path.join(res_output_dir,mname,'DM_predictions.pkl'))
        innerTS = list(predictions.keys())

        for itvalue in np.array(innerTS):
            # loop over task
            fig_output_dir = os.path.join(res_output_dir,mname,'participants_predictions')
            checkdir(fig_output_dir)
            
            for subid in subid_list:
                print(f"plotting {subid}")
                subnum = subid.replace("sub","")
                figs, axes = plt.subplots(5,1, figsize=(11,3*5)) # 5=4+1
                for taskname in ["piratenavigation","localizer"]:
                    run_list = [1] if taskname=="localizer" else [1,2,3,4]
                    n_TR = 326 if taskname == "localizer" else 296
                    for run in run_list:
                        ax = axes[run-1] if taskname=="piratenavigation" else axes[4+run-1]
                        pred_y = predictions[itvalue][taskname][subid][run-1]

                        # transform visual angle to x-y coordinates according to the following formula:
                        # D = D0 * tan(theta) - this is for small angles approximation, where d is the distance from centre on screen, D0 is the distance from eye to screen, theta is the visual angle
                        # note that we do not have the D0 for each participant here, however, this should be the same within each participant
                        # we use this to run the RSA analysis later, which should be invariant to this scaling
                        # here we directly comupute the tan values as an approximation
                        # first we take the median in each TR, pred_y has shape (n_TR, n_innerTR_timesteps, 2)
                        pred_y_median = np.median(pred_y, axis=1)
                        # then we compute the tan values
                        pred_y_tan = np.tan(np.deg2rad(pred_y_median))

                        # plot x and y coordinates as a function of time in the same lineplot
                        timepoints = np.arange(n_TR)+1
                        sns.lineplot(x=timepoints, y=pred_y_tan[:,0], label='X coordinate (tan values)', ax=ax)
                        sns.lineplot(x=timepoints, y=pred_y_tan[:,1], label='Y coordinate (tan values)',ax=ax)
                        ax.set_xlabel('TRs')
                        ax.set_xlim([-1, 327]) # go for the largest n_TR to have same x-axis across plots
                        ax.set_ylabel('Coordinate (tan values)')
                        ax.set_title('%s - run %d'%(taskname, run))

                figs.suptitle('Predicted Eye Coordinates (Median over %d subsamples Per TR) - %s'%(itvalue,subid), y = 0.985, fontsize=12)
                figs.tight_layout()
                figs.savefig(os.path.join(fig_output_dir,'innerTS%d_%s.png'%(itvalue, subid)))
                plt.close(figs)

compute_lzerlag_correlation = True
if compute_lzerlag_correlation:
    
    itvalue = 10  # only one available here as we are using pretrained models
    plt_save_dir = os.path.join(res_output_dir,'localizer_lag_correlation')
    fmridata_dir_win =  "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/"
    beh_dir = os.path.join(fmridata_dir_win,'beh')
    taskname = "localizer"
    n_TR = 326 if taskname == "localizer" else 296
    run_list = [1] if taskname=="localizer" else [1,2,3,4]
    fmri_TR = pirate_defaults["fmri"]["tr"]
    
    print("=================================================== Plotting Lagged Correlations ===============================================")
    
    # plot for each participant
    lag_correlation_dfs = []
    for mname in [x for x in pretrained_models.keys() if x != 'm6_openclosed']:
        predictions = load(os.path.join(res_output_dir,mname,'DM_predictions.pkl'))
        for run in run_list:
            for subid in subid_list:
                print(f"processing lag correlation for {subid} in run {run} using model {mname}", end='\r',flush=True)
                # load beh data
                tmpfn = os.path.join(beh_dir, subid ,'sub-%s_task-%s_run-%d_onsets.csv' % (subid.replace("sub",""), taskname,run))
                beh_data = pd.read_csv(tmpfn)
                onset_stim = beh_data["onset_stimuli"].values
                dura_stim  = beh_data["duration_stimuli"].values
            
                # load model predictions
                pred_y = predictions[itvalue][taskname][subid][run_list.index(run)] # shape (n_TR, itvalue, 2)
                for plottype in ["median","subTRsamples"]:
                    if plottype == "median":
                        pred_y_plot = np.median(pred_y, axis=1) # shape (n_TR, 2)
                        pred_y_tan = np.tan(np.deg2rad(pred_y_plot))
                        # create stimulus location timecourse
                        timseries, stimloc_timecourse = create_event_timecourse(onset_stim, dura_stim, beh_data[["stim_x", "stim_y"]].to_numpy(), n_TR, fmri_TR, inner_timesteps=1)
                        # compute correlations with lags
                        lags, correlations = compute_corr_with_lags(stimloc_timecourse, pred_y_tan, max_lag=10)
                    elif plottype == "subTRsamples":
                        pred_y_plot = pred_y.reshape(-1,2)  # shape (n_TR*itvalue, 2)
                        pred_y_tan = np.tan(np.deg2rad(pred_y_plot))
                        # create stimulus location timecourse
                        timseries, stimloc_timecourse = create_event_timecourse(onset_stim, dura_stim, beh_data[["stim_x", "stim_y"]].to_numpy(), n_TR, fmri_TR, inner_timesteps=itvalue)
                        # compute correlations with lags
                        lags, correlations = compute_corr_with_lags(stimloc_timecourse, pred_y_tan, max_lag=10*itvalue)
                    
                    # do the plot
                    plt_savename = f"{mname[:2]}_{subid}_{plottype}.png"                    
                    fig, axes = plt.subplots(3,1,figsize=(14,6))

                    # plot stimulus location timecourse in the first subplot
                    sns.lineplot(x=timseries,y=stimloc_timecourse[:,0],label="stimulus location x",ax=axes[0],color='orange',alpha=1)
                    sns.lineplot(x=timseries,y=stimloc_timecourse[:,1],label="stimulus location y",ax=axes[0],color='blue',alpha=1)
                    axes[0].legend(ncol=1, loc='upper right')
                    axes[0].set_title("Stimulus Location")

                    # plot predicted gaze location timecourse in the second subplot
                    sns.lineplot(x=timseries,y=pred_y_tan[:,0],label="pred gaze x",ax=axes[1],color='orange')
                    sns.lineplot(x=timseries,y=pred_y_tan[:,1],label="pred gaze y",ax=axes[1],color='blue')
                    axes[1].legend(ncol=1, loc='upper right')
                    axes[1].set_title(f"Predicted Gaze Location")
                    for ax in axes:
                        ax.set_xlabel("Time (s)")

                    # plot correlations vs lags in the third subplot
                    # get the unit of lags in seconds
                    if plottype == "median":
                        # unit is fmri_TR, and we plot with TR unit
                        plot_lags = lags
                        xlabel = "Lag (TR)"
                        lag_TR = lags
                    elif plottype == "subTRsamples":
                        plot_lags = lags * fmri_TR / itvalue
                        xlabel = "Lag (s)"
                        lag_TR = lags / itvalue
                    lags_sec = lag_TR  * fmri_TR
                    axes[2], max_corr_lag_x, max_corr_lag_y = plot_corr_with_lags(plot_lags,correlations,xlabel=xlabel,ax=axes[2], max_annot = "text")

                    fig.suptitle(f"Localizer task: Stimulus location vs Predicted Gaze Position\n{mname}\n {plottype} prediction",fontsize=13)
                    plt.tight_layout()
                    fig.savefig(os.path.join(plt_save_dir, plt_savename))
                    plt.close(fig)

                    tmpdf = pd.DataFrame({
                        "lag_TR": lag_TR,
                        "lag_sec": lags_sec,
                        "correlation_x": correlations[:,0],
                        "correlation_y": correlations[:,1]
                    }).assign(
                        subid=subid,
                        run=run,
                        pretrainedmodel=mname,
                        taskname=taskname,
                        max_corr_lag_x=max_corr_lag_x,
                        max_corr_lag_y=max_corr_lag_y,
                        compute_method=plottype,
                        meanacc = beh_data["acc"].mean(),
                        meanrt=(beh_data["onset_response"] - beh_data["onset_stimuli"]).mean()
                    )
                    lag_correlation_dfs.append(tmpdf)

    lag_correlation_df = pd.concat(lag_correlation_dfs, ignore_index=True).reset_index(drop=True)
    lag_correlation_df.to_csv(os.path.join(plt_save_dir,'lag_correlations_allsubs.csv'), index=False)


                    