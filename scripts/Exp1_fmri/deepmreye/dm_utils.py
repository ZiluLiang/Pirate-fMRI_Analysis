import numpy as np
import os
from zpyhelper.filesys import checkdir
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Import DeepMReye
from deepmreye import preprocess, train, analyse,architecture
from deepmreye.util import model_opts, data_generator

from joblib import dump, load, Parallel, delayed

project_path = '/home/liangz/pirate_fmri/Analysis'
sys.path.append(project_path)
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silent TF GPU warnings

def infer_gaze_position(preproc_data_fn, inner_timesteps, pretrained_model_fn):
    # Load data: The following two approaches are the same
    #subdata = np.load(subfp,mmap_mode='r')
    #X = np.array([subdata[f'data_{tr}'] for tr in n_TR])[...,np.newaxis]
    X, _ = data_generator.get_all_subject_data(preproc_data_fn)          

    # Load pretrained model      
    opts = model_opts.get_opts()
    opts["inner_timesteps"] = inner_timesteps
    _, model_inference = architecture.create_standard_model(X.shape[1::], opts)
    model_inference.load_weights(pretrained_model_fn)

    # Run inference on loaded data
    (pred_y, _) = model_inference.predict(X)
    return pred_y

def corr_significance_label(x, y, return_r=False):
        r, p = pearsonr(x, y)
        r_str = "%.3f" % r
        if return_r:
            return r, p
        else:
            if p < 0.001:
                return f'{r_str}***'
            elif p < 0.01:
                return f'{r_str}**'
            elif p < 0.05:
                return f'{r_str}*'
            else:
                return f'{r_str}ns'
            
def compute_corr_with_lags(event_timecourse: np.ndarray, gaze_timecourse: np.ndarray, max_lag:int=10)->tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation between event_timecourse and pred_y with lags from -max_lag to +max_lag

    Parameters:
    ============
    event_timecourse: np.ndarray
        shape (n_TR, 2), event timecourse (e.g. stimulus location or response location)
    gaze_timecourse: np.ndarray
        shape (n_TR, 2), predicted gaze location timecourse
    max_lag: int
        maximum lag to compute correlation for, unit of lag depends on unt unit of the timecourse of event_timecourse and pred_y (e.g., if both are in TRs, then lag is in TRs)
    Returns:
    ============
    lags: np.ndarray
        shape (n_lags,), lags from -max_lag to +max_lag
    correlations: np.ndarray
        shape (n_lags, 2), correlations for x and y at each lag
    """
    n_timepoints = event_timecourse.shape[0]
    lags = np.arange(-max_lag, max_lag+1)
    correlations = []
    for lag in lags:
        if lag < 0:
            ev_shifted = event_timecourse[-lag:,:]
            gaze_shifted = gaze_timecourse[:n_timepoints+lag,:]
        elif lag > 0:
            ev_shifted = event_timecourse[:n_timepoints-lag,:]
            gaze_shifted = gaze_timecourse[lag:n_timepoints,:]
        else:
            ev_shifted = event_timecourse
            gaze_shifted = gaze_timecourse
        # compute correlation for x and y separately, ignoring nan values
        valid_idx_x = ~np.isnan(ev_shifted[:,0])
        valid_idx_y = ~np.isnan(ev_shifted[:,1])
        corr_x, _ = corr_significance_label(ev_shifted[:,0][valid_idx_x], gaze_shifted[:,0][valid_idx_x], return_r=True)
        corr_y, _ = corr_significance_label(ev_shifted[:,1][valid_idx_y], gaze_shifted[:,1][valid_idx_y], return_r=True)
        correlations.append((corr_x, corr_y))
    return lags, np.array(correlations)  # shape (n_lags, 2)

def create_event_timecourse(event_onset, event_duration, event_value, n_TR=None, TR=None, inner_timesteps=1, timseries=None)->np.ndarray:
    """
    Create a timecourse of events given their onsets and durations. \n
    When an event is present, the corresponding timepoints in the timecourse will be filled with the event_value, otherwise will be filled with nan.

    Parameters:
    ============
    event_onset: np.ndarray
        shape (n_events,), onset times of events in seconds
    event_duration: np.ndarray
        shape (n_events,), duration times of events in seconds
    event_value: np.ndarray
        shape (n_events, n_features), values associated with each event
    n_TR: int
        number of TRs in the timecourse
    TR: float
        TR duration in seconds
    inner_timesteps: int
        number of inner timesteps per TR (default is 1, meaning no inner timesteps), will create timecourse with n_TR * inner_timesteps timepoints
        timeseries will always be in units of seconds 
    timseries: np.ndarray or None
        shape (n_timepoints,), custom time series points in seconds. If None, will be generated based on n_TR, TR, and inner_timesteps.
        if the provided timseries length does not match n_TR * inner_timesteps, a ValueError will be raised.

    Returns:
    ============
    timseries: np.ndarray
        shape (n_timepoints,), time series points in seconds
    event_timecourse: np.ndarray
        shape (n_timepoints, event_value.shape[1]), timecourse of events with event values and nan for absence
    """
    assert all([timseries is None, not (n_TR is None or TR is None)]) or timseries is not None, "Either provide timseries or both n_TR and TR"
    if timseries is None:
        if inner_timesteps == 1:
            timseries = (np.arange(n_TR)+1) * TR
        else:
            timseries = (np.arange(n_TR*inner_timesteps)+1) * (TR/inner_timesteps)
    else:
        if n_TR is not None and TR is not None:
            expected_length = n_TR * inner_timesteps
            if len(timseries) != expected_length:
                raise ValueError(f"Provided timseries length {len(timseries)} does not match expected length {expected_length} based on n_TR and inner_timesteps.")

    event_timecourse = np.full((timseries.shape[0],event_value.shape[1]), np.nan)
    # filter out the nan event onsets
    valid_events = ~np.isnan(event_onset)
    event_onset = event_onset[valid_events]
    event_duration = event_duration[valid_events]
    event_value = event_value[valid_events]
    for onset, duration, value in zip(event_onset, event_duration, event_value):
        start_idx = np.searchsorted(timseries, onset, side='left')
        end_idx = np.searchsorted(timseries, onset + duration, side='right')
        if start_idx < end_idx: 
            event_timecourse[start_idx:end_idx] = value
        elif start_idx == end_idx:
            event_timecourse[start_idx] = value
        else:
            ValueError("Event onset and duration result in invalid timecourse assignment.")
    return timseries, event_timecourse


def plot_corr_with_lags(plot_lags,correlations,xlabel="Lags",ax=None, max_annot = "text"):
    if ax is None:
        _, ax = plt.subplots(figsize=(4,2))
    
    sns.lineplot(x=plot_lags,y=correlations[:,0],label="correlation x",ax=ax,color='orange')
    sns.lineplot(x=plot_lags,y=correlations[:,1],label="correlation y",ax=ax,color='blue')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pearson's r")
    ax.legend(ncol=1, loc='upper right')
    # find the lag with the maximum correlation for x and y
    max_corr_idx_x = np.argmax(correlations[:,0])
    max_corr_idx_y = np.argmax(correlations[:,1])
    max_corr_lag_x = plot_lags[max_corr_idx_x]
    max_corr_lag_y = plot_lags[max_corr_idx_y]
    # mark the maximum correlation points using vertical lines
    ax.axvline(x=max_corr_lag_x, color='orange', linestyle='--')
    ax.axvline(x=max_corr_lag_y, color='blue', linestyle='--')
    if max_annot == "ticks":
        curr_xticks = list(ax.get_xticks())
        if np.abs(curr_xticks-max_corr_lag_x).min() > 1e-1:
            curr_xticks.append(max_corr_lag_x)
        if np.abs(curr_xticks-max_corr_lag_y).min() > 1e-1:
            curr_xticks.append(max_corr_lag_y)
        curr_xticks = np.array(sorted(curr_xticks))
        ax.set_xticks(curr_xticks)
    elif max_annot == "text":
        # else add at the bottom of the plot
        # if x and y lags are too close, offset the text a bit
        if np.abs(max_corr_lag_x - max_corr_lag_y) < 0.5:
            offset = 0.3
        else:
            offset = 0.0
        ax.text(max_corr_lag_x, ax.get_ylim()[0] + offset, f"{max_corr_lag_x:.2f}", color='orange', ha='left', va='bottom')
        ax.text(max_corr_lag_y, ax.get_ylim()[0], f"{max_corr_lag_y:.2f}", color='blue', ha='left', va='bottom')
    ax.set_title(f"Lagged Correlations\nMax Correlation:  X -{correlations[max_corr_idx_x,0]:.2f}, Y - {correlations[max_corr_idx_y,1]:.2f}")
    return ax, max_corr_lag_x, max_corr_lag_y