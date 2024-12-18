import numpy as np
import pandas as pd
import scipy
from zpyhelper.MVPA.rdm import lower_tri

def generate_filters(stimdf,verbose=False)->dict:
    """Generate filters for picking specific subset of the data

    Parameters
    ----------
    stimdf : pandas data frame
        dataframe for the stimuli
    verbose : bool, optional
        print the size of the subset of data, by default False

    Returns
    -------
    dictionary
        a dictionary of filters for the navigation task with they following keys:\n
        "training_nocenter": training stimuli that are not in the screen center (8 in total)\n
        "training_all":      all training stimuli(9 in total)\n
        "test":              all test stimuli (16 in total)\n
        "allstim_nocenter":  all stimuli that are not in the screen center (24 in total)\n
        "allstim":           all stimuli 
    """
    filters = {
        "training_nocenter":[np.logical_and(np.logical_xor(x==0,y==0),t=="navigation")  for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "training_all":     [np.logical_and(np.logical_or(x==0,y==0),t=="navigation")   for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "test":             [np.logical_and(~np.logical_or(x==0,y==0),t=="navigation")  for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "allstim_nocenter": [np.logical_and(~np.logical_and(x==0,y==0),t=="navigation") for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "allstim":          [t=="navigation" for t in stimdf["taskname"].to_numpy()]
    }
    if verbose:
        for k,v in filters.items():
            print(f"{k}: size = {sum(v)}")
    return filters

def minmax_scale(rdm_unscaled:np.ndarray,newmin:float=0,newmax:float=1)->np.ndarray:
    """rescale rdm values to a new range, note that only the lower triangular part is used to calculate the rescaling, on-diagonal will be set to new minimum

    Parameters
    ----------
    rdm_unscaled : np.ndarray
        input rdm
    newmin : float, optional
        new minimum, by default 0
    newmax : float, optional
        new maximum, by default 1

    Returns
    -------
    np.ndarray
        scaled rdm
    """
    rdm_unscaledval = lower_tri(rdm_unscaled)[0]    
    minv,maxv = rdm_unscaledval.min(),rdm_unscaledval.max()
    #new range
    rdm_scaled_val = newmin + (newmax - newmin) * (rdm_unscaledval - minv) / (maxv - minv)
    scaledrdm = np.zeros_like(rdm_unscaled)
    scaledrdm[lower_tri(scaledrdm)[1]] = rdm_scaled_val
    scaledrdm = scaledrdm + scaledrdm.T
    np.fill_diagonal(scaledrdm,newmin)
    return scaledrdm


def parallel_axes_cosine_sim(xstims,ystims,return_codingdirs = False):
    """compute the cosine similarity between the coding directions (of the same pair of axis locations) from different training axes.\n
    For example, to obtain the coding directions pointing from axis location of -2 to -1, we need to retrieve the activity patterns of the stimuli from the corresponding training locations. They are: \n
    (1) on x axis, activity patterns of [-2,0] and [-1,0]\n
    (2) on y axis, activity patterns of [0,-2] and [0,-1]\n
    Then, we compute the coding directions of x and y axis as the difference between the stimuli pair from the corresponding axis:\n
    codingdirection_x = activitypattern_of_[-2,0] - activitypattern_of_[-1,0]\n
    codingdirection_y = activitypattern_of_[0,-2] - activitypattern_of_[0,-1]\n
    Finally, we compute the cosine similarity between these two coding directions.

    This process is repeated for all the possible axis locations pairs, and all the cosine similarities are returned in a symmetric nloc*nloc matrix.\n
    The upper triagular entries of this matrix are the cosine similarity of C(nloc,2) unique pairs. All the other entries are set to np.nan


    Parameters
    ----------
    xstims : np.ndarray
        activity patterns of the training stimuli on x axis. Each row is the activity pattern of a training stimulus, ordered by axis location. This order must be the same as the one of `ystim`.
    ystims : np.ndarray
        activity patterns of the training stimuli on y axis. Each row is the activity pattern of a training stimulus, ordered by axis location. This order must be the same as the one of `ystim`.
    return_codingdirs: bool
        Flag for returning the matrices of coding directions. If true, the matrices corresponding to coding directions on x and y will be returned

    Returns
    -------
    np.ndarray
        matrix of cosine similarities. If xstims and ystims each has nloc rows, the matrix will be of shape (nloc,nloc).\n
        The upper triagular entries of this matrix are the cosine similarity of C(nloc,2) unique pairs. All the other entries are set to np.nan
    or a tuple of (cosine similarities, coding directions on x, coding directions on y)
    """
    assert xstims.shape[0] == ystims.shape[0]
    assert xstims.shape[1] == ystims.shape[1]
    nloc = xstims.shape[0]
    nf = xstims.shape[1]
    x_coding_dirs = np.full((nloc,nloc,nf),fill_value=np.nan)
    y_coding_dirs = np.full((nloc,nloc,nf),fill_value=np.nan)
    xycosinesim   = np.full((nloc,nloc),fill_value=np.nan)
    for idx1 in range(nloc):
        for idx2 in range(nloc):
            if idx1 < idx2:
                x_coding_dirs[idx1,idx2] = xstims[idx1] - xstims[idx2]
                y_coding_dirs[idx1,idx2] = ystims[idx1] - ystims[idx2]
                xycosinesim[idx1,idx2] = 1-scipy.spatial.distance.cosine(x_coding_dirs[idx1,idx2],y_coding_dirs[idx1,idx2])
    if return_codingdirs:
        return xycosinesim, x_coding_dirs, y_coding_dirs
    else:
        return xycosinesim