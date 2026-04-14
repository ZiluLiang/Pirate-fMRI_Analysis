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
decoding_res_dir = f"/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/deepmreye/results/{raw_img_type}/"


itvalue = 10  # only one available here
fmridata_dir_win =  "/mnt/e/pirate_fmri/Analysis/data/Exp1_fmri/fmri/"
beh_dir = os.path.join(fmridata_dir_win,'beh')
taskname = "piratenavigation"
n_TR = 326 if taskname == "localizer" else 296
run_list = [1] if taskname=="localizer" else [1,2,3,4]
fmri_TR = pirate_defaults["fmri"]["tr"]
mname = "m7_multiDS1to5" # this is the winning model selected from inspecting localizer task gaze prediction

#%%
compute_lagcorr_wholetimecourse = False
if compute_lagcorr_wholetimecourse:
    print("=================================================== Computing Lagged Correlations ===============================================\n")
    # here we compute lagged correlations between predicted gaze position and stimulus/response locations for each participant and run using the whole timecourse

    # plot for each participant
    lagcorr_output_dir = os.path.join(decoding_res_dir,'treasurehunt_lag_correlation')
    checkdir(lagcorr_output_dir)

    lag_correlation_dfs = []
    predictions = load(os.path.join(decoding_res_dir,mname,'DM_predictions.pkl'))
    for run in run_list:
        for subid in subid_list:
            print(f"processing lag correlation for {subid} in run {run} using model {mname}", end='\r',flush=True)
            # load beh data
            tmpfn = os.path.join(beh_dir, subid ,'sub-%s_task-%s_run-%d_onsets.csv' % (subid.replace("sub",""), taskname,run))
            beh_data = pd.read_csv(tmpfn)
            onset_stim = beh_data["onset_stimuli"].values
            dura_stim  = beh_data["duration_stimuli"].values

            onset_respstart = beh_data["onset_response"].values
            onset_respend = beh_data["onset_submit"].values
            dura_resp  = beh_data["duration_response"].values

            start_locations = beh_data[["start_x","start_y"]].to_numpy()
            response_locations = beh_data[["resp_x","resp_y"]].to_numpy()
            stimuli_locations = beh_data[["stim_x","stim_y"]].to_numpy()
            # load model predictions
            pred_y = predictions[itvalue][taskname][subid][run_list.index(run)] # shape (n_TR, itvalue, 2)
            
            for compute_method in ["median","subTRsamples"]:
                if compute_method == "median": # sampling at fmri_TR resolution, so we take median across sub-TR samples
                    pred_y_plot = np.median(pred_y, axis=1) # shape (n_TR, 2)
                    pred_y_tan = np.tan(np.deg2rad(pred_y_plot))
                    # create stimulus and response location timecourse
                    timseries, stim_timecourse = create_event_timecourse(onset_stim, dura_stim, stimuli_locations, n_TR, fmri_TR, inner_timesteps=1)
                    # assign a stick duration to response onset/submit events to model a discrete event
                    shortdur = np.zeros_like(dura_resp)
                    timseries, respst_timecourse = create_event_timecourse(onset_respstart, shortdur, start_locations, n_TR, fmri_TR, inner_timesteps=1)
                    timseries, resped_timecourse = create_event_timecourse(onset_respend, shortdur, response_locations, n_TR, fmri_TR, inner_timesteps=1)

                    # compute lagged correlations: 
                    lags_stim, corrs_stim = compute_corr_with_lags(stim_timecourse, pred_y_tan, max_lag=10)
                    lags_respst, corrs_respst = compute_corr_with_lags(respst_timecourse, pred_y_tan, max_lag=10)
                    lags_resped, corrs_resped = compute_corr_with_lags(resped_timecourse, pred_y_tan, max_lag=10)
                    assert np.array_equal(lags_stim, lags_respst)
                    assert np.array_equal(lags_stim, lags_resped)
                    lags = lags_stim
                elif compute_method == "subTRsamples": # sampling at sub-TR resolution, so we reshape the predictions
                    pred_y_plot = pred_y.reshape(-1,2)  # shape (n_TR*itvalue, 2)
                    pred_y_tan = np.tan(np.deg2rad(pred_y_plot))
                    # create stimulus and response location timecourse
                    timseries, stim_timecourse = create_event_timecourse(onset_stim, dura_stim, stimuli_locations, n_TR, fmri_TR, inner_timesteps=itvalue)
                    timseries, respst_timecourse = create_event_timecourse(onset_respstart, shortdur, start_locations, n_TR, fmri_TR, inner_timesteps=itvalue)
                    timseries, resped_timecourse = create_event_timecourse(onset_respend, shortdur, response_locations, n_TR, fmri_TR, inner_timesteps=itvalue)
                    # compute lagged correlations
                    lags_stim, corrs_stim = compute_corr_with_lags(stim_timecourse, pred_y_tan, max_lag=10*itvalue)
                    lags_respst, corrs_respst = compute_corr_with_lags(respst_timecourse, pred_y_tan, max_lag=10*itvalue)
                    lags_resped, corrs_resped = compute_corr_with_lags(resped_timecourse, pred_y_tan, max_lag=10*itvalue)
                    assert np.array_equal(lags_stim, lags_respst)
                    assert np.array_equal(lags_stim, lags_resped)
                    lags = lags_stim

                # assume that when things are not present should look at (0,0), this is just for plotting, otherwise events are too sparse and will make the plot messy
                stim_timecourse[np.isnan(stim_timecourse)] = 0
                respst_timecourse[np.isnan(respst_timecourse)] = 0
                resped_timecourse[np.isnan(resped_timecourse)] = 0
                # do the plot
                plt_savename = f"{mname[:2]}_{subid}_{run}_{compute_method}.png"                    
                fig, axes = plt.subplots(5,1,figsize=(14,10))

                # plot stimulus and response location timecourse in the first subplot
                locations = {
                    "stimulus": stim_timecourse,
                    "resp starting": respst_timecourse,
                    "resp final": resped_timecourse
                }
                linestyles = {
                    "stimulus": "dotted",
                    "resp starting": "dashed",
                    "resp final": "solid"
                }
                colors = {
                    "stimulus": ("salmon","turquoise"),
                    "resp starting": ("darkorange","dodgerblue"),
                    "resp final": ("orangered","steelblue")
                }
                for loc_label, loc_timecourse in locations.items():
                    sns.lineplot(x=timseries,y=loc_timecourse[:,0],label=f"{loc_label} location x",ax=axes[0],color=colors[loc_label][0],alpha=0.6, linestyle=linestyles.get(loc_label, '-'))
                    sns.lineplot(x=timseries,y=loc_timecourse[:,1],label=f"{loc_label} location y",ax=axes[0],color=colors[loc_label][1],alpha=0.6, linestyle=linestyles.get(loc_label, '-'))
                axes[0].legend(ncol=1, loc='upper right',fontsize='small')
                axes[0].set_title("Stimulus, Starting and Final Response Locations")

                # plot predicted gaze location timecourse in the second subplot
                sns.lineplot(x=timseries,y=pred_y_tan[:,0],label="pred gaze x",ax=axes[1],color='orange')
                sns.lineplot(x=timseries,y=pred_y_tan[:,1],label="pred gaze y",ax=axes[1],color='blue')
                axes[1].legend(ncol=1, loc='upper right')
                axes[1].set_title(f"Predicted Gaze Location")
                for ax in axes:
                    ax.set_xlabel("Time (s)")

                # plot correlations vs lags in the third and fourth subplot
                # get the unit of lags in seconds
                if compute_method == "median":
                    # unit is fmri_TR, and we plot with TR unit
                    plot_lags = lags
                    xlabel = "Lag (TR)"
                    lag_TR = lags
                elif compute_method == "subTRsamples":
                    plot_lags = lags * fmri_TR / itvalue
                    xlabel = "Lag (s)"
                    lag_TR = lags / itvalue
                lags_sec = lag_TR  * fmri_TR
                axes[2], max_stimcorr_lag_x, max_stimcorr_lag_y = plot_corr_with_lags(plot_lags,corrs_stim,xlabel=xlabel,ax=axes[2])
                axes[2].set_title("Lagged Correlations: Stimulus Location vs Predicted Gaze Position")
                axes[3], max_respstcorr_lag_x, max_respstcorr_lag_y = plot_corr_with_lags(plot_lags,corrs_respst,xlabel=xlabel,ax=axes[3])
                axes[3].set_title("Lagged Correlations: Response Start Location vs Predicted Gaze Position")
                axes[4], max_respedcorr_lag_x, max_respedcorr_lag_y = plot_corr_with_lags(plot_lags,corrs_resped,xlabel=xlabel,ax=axes[4])
                axes[4].set_title("Lagged Correlations: Response End Location vs Predicted Gaze Position")
                fig.suptitle(f"Treasure Hunt task Run {run}: Stimulus/Response location vs Predicted Gaze Position\n{mname}")
                plt.tight_layout()
                fig.savefig(os.path.join(lagcorr_output_dir, plt_savename))
                plt.close(fig)

                tmpdf = pd.DataFrame({
                    "lag_TR": lag_TR,
                    "lag_sec": lags_sec,
                    "stimcorr_x": corrs_stim[:,0],
                    "stimcorr_y": corrs_stim[:,1],
                    "respstcorr_x": corrs_respst[:,0],
                    "respstcorr_y": corrs_respst[:,1],
                    "respedcorr_x": corrs_resped[:,0],
                    "respedcorr_y": corrs_resped[:,1],
                }).assign(
                    subid=subid,
                    run=run,
                    pretrainedmodel=mname,
                    taskname=taskname,
                    max_stimcorr_lag_x=max_stimcorr_lag_x,
                    max_stimcorr_lag_y=max_stimcorr_lag_y,
                    max_respstcorr_lag_x=max_respstcorr_lag_x,
                    max_respstcorr_lag_y=max_respstcorr_lag_y,
                    max_respedcorr_lag_x=max_respedcorr_lag_x,
                    max_respedcorr_lag_y=max_respedcorr_lag_y,
                    compute_method=compute_method,
                )
                lag_correlation_dfs.append(tmpdf)


    lag_correlation_df = pd.concat(lag_correlation_dfs, ignore_index=True).reset_index(drop=True)
    lag_correlation_df.to_csv(os.path.join(lagcorr_output_dir,'lag_correlations_allsubs.csv'), index=False)

#%%
compute_lagcorr_splittrialtypes = True
if compute_lagcorr_splittrialtypes:
    print("=================================================== Computing Lagged Correlations (Split Trial types) ===============================================\n")
    # here we compute lagged correlations between predicted gaze position and stimulus/response locations for each participant and run using split trials

    # plot for each participant
    lagcorrst_output_dir = os.path.join(decoding_res_dir,'treasurehunt_lag_correlation_splittrialtypes')
    checkdir(lagcorrst_output_dir)
    
    splitttype_lag_correlation_dfs = []
    predictions = load(os.path.join(decoding_res_dir,mname,'DM_predictions.pkl'))
    for run in run_list:
        for subid in subid_list:
            print(f"processing lag correlation for {subid} in run {run} using model {mname}", end='\r',flush=True)
            # load beh data
            tmpfn = os.path.join(beh_dir, subid ,'sub-%s_task-%s_run-%d_onsets.csv' % (subid.replace("sub",""), taskname,run))
            beh_data = pd.read_csv(tmpfn)
            onset_stim = beh_data["onset_stimuli"].values
            dura_stim  = beh_data["duration_stimuli"].values

            onset_respstart = beh_data["onset_response"].values
            onset_respend = beh_data["onset_submit"].values
            dura_resp  = beh_data["duration_response"].values

            # compute mean duration of response phase
            mean_dura_resp = np.nanmean(dura_resp)
            mean_dura_stim = np.nanmean(dura_stim)
            # filter out trials with/without responses
            resp_trial_mask = ~np.isnan(onset_respstart)
            trial_filters = {
                "with_response": resp_trial_mask,
                "without_response": ~resp_trial_mask
            }


            start_locations = beh_data[["start_x","start_y"]].to_numpy()
            response_locations = beh_data[["resp_x","resp_y"]].to_numpy()
            stimuli_locations = beh_data[["stim_x","stim_y"]].to_numpy()
            # load model predictions
            pred_y = predictions[itvalue][taskname][subid][run_list.index(run)] # shape (n_TR, itvalue, 2)
            
            for compute_method in ["median","subTRsamples"]:
                for ttype, tfilt in trial_filters.items():
                    if compute_method == "median": # sampling at fmri_TR resolution, so we take median across sub-TR samples
                        pred_y_plot = np.median(pred_y, axis=1) # shape (n_TR, 2)
                        pred_y_tan = np.tan(np.deg2rad(pred_y_plot))
                        # create stimulus and response location timecourse
                        timseries, stim_timecourse = create_event_timecourse(onset_stim[tfilt], dura_stim[tfilt], stimuli_locations[tfilt], n_TR, fmri_TR, inner_timesteps=1)

                        # compute lagged correlations: 
                        lags, corrs_stim = compute_corr_with_lags(stim_timecourse, pred_y_tan, max_lag=10)
                        
                    elif compute_method == "subTRsamples": # sampling at sub-TR resolution, so we reshape the predictions
                        pred_y_plot = pred_y.reshape(-1,2)  # shape (n_TR*itvalue, 2)
                        pred_y_tan = np.tan(np.deg2rad(pred_y_plot))
                        # create stimulus and response location timecourse
                        timseries, stim_timecourse = create_event_timecourse(onset_stim[tfilt], dura_stim[tfilt], stimuli_locations[tfilt], n_TR, fmri_TR, inner_timesteps=itvalue)
                        # compute lagged correlations
                        lags, corrs_stim = compute_corr_with_lags(stim_timecourse, pred_y_tan, max_lag=10*itvalue)
                        

                    # assume that when things are not present should look at (0,0), this is just for plotting, otherwise events are too sparse and will make the plot messy
                    stim_timecourse[np.isnan(stim_timecourse)] = 0
                    # do the plot
                    plt_savename = f"{mname[:2]}_{subid}_{run}_{ttype}_{compute_method}.png"                    
                    fig, axes = plt.subplots(3,1,figsize=(14,6))

                    # plot stimulus and response location timecourse in the first subplot
                    locations = {
                        "stimulus": stim_timecourse,
                    }
                    linestyles = {
                        "stimulus": "dotted",
                    }
                    colors = {
                        "stimulus": ("salmon","turquoise"),
                    }
                    for loc_label, loc_timecourse in locations.items():
                        sns.lineplot(x=timseries,y=loc_timecourse[:,0],label=f"{loc_label} location x",ax=axes[0],color=colors[loc_label][0],alpha=1, linestyle=linestyles.get(loc_label, '-'))
                        sns.lineplot(x=timseries,y=loc_timecourse[:,1],label=f"{loc_label} location y",ax=axes[0],color=colors[loc_label][1],alpha=1, linestyle=linestyles.get(loc_label, '-'))
                    axes[0].legend(ncol=1, loc='upper right',fontsize='small')
                    axes[0].set_title("Stimulus and Final Response Locations")

                    # plot predicted gaze location timecourse in the second subplot
                    sns.lineplot(x=timseries,y=pred_y_tan[:,0],label="pred gaze x",ax=axes[1],color='orange')
                    sns.lineplot(x=timseries,y=pred_y_tan[:,1],label="pred gaze y",ax=axes[1],color='blue')
                    axes[1].legend(ncol=1, loc='upper right')
                    axes[1].set_title(f"Predicted Gaze Location")
                    for ax in axes:
                        ax.set_xlabel("Time (s)")

                    # plot correlations vs lags in the third and fourth subplot
                    # get the unit of lags in seconds
                    if compute_method == "median":
                        # unit is fmri_TR, and we plot with TR unit
                        plot_lags = lags
                        xlabel = "Lag (TR)"
                        lag_TR = lags
                    elif compute_method == "subTRsamples":
                        plot_lags = lags * fmri_TR / itvalue
                        xlabel = "Lag (s)"
                        lag_TR = lags / itvalue
                    lags_sec = lag_TR  * fmri_TR
                    axes[2], max_stimcorr_lag_x, max_stimcorr_lag_y = plot_corr_with_lags(plot_lags,corrs_stim,xlabel=xlabel,ax=axes[2])
                    axes[2].set_title("Lagged Correlations: Stimulus Location vs Predicted Gaze Position")
                    
                    fig.suptitle(f"Run {run} Trials {ttype.replace('_','')}: Stimulus/Response location vs Predicted Gaze Position\n{mname}")
                    plt.tight_layout()
                    fig.savefig(os.path.join(lagcorrst_output_dir, plt_savename))
                    plt.close(fig)

                    tmpdf = pd.DataFrame({
                        "lag_TR": lag_TR,
                        "lag_sec": lags_sec,
                        "stimcorr_x": corrs_stim[:,0],
                        "stimcorr_y": corrs_stim[:,1],
                        "trial_type": ttype,
                    }).assign(
                        subid=subid,
                        run=run,
                        pretrainedmodel=mname,
                        taskname=taskname,
                        max_stimcorr_lag_x=max_stimcorr_lag_x,
                        max_stimcorr_lag_y=max_stimcorr_lag_y,
                        compute_method=compute_method,
                        meandur_resp = mean_dura_resp,
                        meandur_stim = mean_dura_stim,
                        meandur_s2resp = mean_dura_resp + mean_dura_stim
                    )
                    splitttype_lag_correlation_dfs.append(tmpdf)

    splitttype_lag_correlation_df = pd.concat(splitttype_lag_correlation_dfs, ignore_index=True).reset_index(drop=True)
    splitttype_lag_correlation_df.to_csv(os.path.join(lagcorrst_output_dir,'splittrialtype_lag_correlations_allsubs.csv'), index=False)