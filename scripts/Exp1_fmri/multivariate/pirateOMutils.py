import numpy as np
import pandas as pd
import scipy
import itertools
from sklearn.metrics import r2_score
from zpyhelper.MVPA.rdm import lower_tri
from zpyhelper.MVPA.preprocessors import split_data

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
        "trainloc_nocenter": all the non-center locations the localizer task (8 in total)\n
        "trainloc_all":      all the locations the localizer task (9 in total)\n
    """
    filters = {
        "training_nocenter":[np.logical_and(np.logical_xor(x==0,y==0),t=="navigation")  for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "training_all":     [np.logical_and(np.logical_or(x==0,y==0),t=="navigation")   for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "test":             [np.logical_and(~np.logical_or(x==0,y==0),t=="navigation")  for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "allstim_nocenter": [np.logical_and(~np.logical_and(x==0,y==0),t=="navigation") for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "allstim":          [t=="navigation" for t in stimdf["taskname"].to_numpy()],
        "trainloc_nocenter":[np.logical_and(np.logical_xor(x==0,y==0),t=="localizer")  for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()],
        "trainloc_all":     [np.logical_and(np.logical_or(x==0,y==0),t=="localizer")   for x,y,t in stimdf[["stim_x","stim_y","taskname"]].to_numpy()]
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
    
def cross_val_SVD(X,session_labels,flag_centering=False):
    """Compute the optimal number of components for each session using cross-validation
    Each session is left out in turn, the remaining data is further split into training data and validation data.
    We perform SVD on the training data and reconstruct the training data using the first n components.
    The optimal number of components is determined by the highest correlation between the reconstructed training data and the validation data.
    Then, we averaged across the training and validation data matrix, and performed reconstruction on this averaged matrix with the estimated optimal number of components.
    The reconstruction quality is measured by the correlation between the reconstructed data and the test data.

    Assert that each session have equal number of conditions
    Parameters
    ----------
    X : numpy.ndarray
        Activity pattern matrix, shape = [n_samples, n_features]
    session_labels : numpy.ndarray
        session label for each row of x, shape = [n_samples,]
    flag_centering : bool
        whether to center the data before SVD
    
    Returns
    -------
    est_d : numpy.ndarray
        estimated number of components for each session
    reconstruction_corr : numpy.ndarray
        correlation between the reconstructed data and the test data for each session
    unique_sess : numpy.ndarray
        unique session labels
    
    """
    if flag_centering:
        X = X-np.mean(X,axis=0) # center the data

    unique_sess = np.unique(session_labels)
    assert unique_sess.size>3, "Number of unique sessions must be greater than 3 to perform proper cross-validation"
    
    # initializer place holders for results
    est_d = np.full_like(unique_sess,fill_value=np.nan)
    reconstruction_corr =  np.full_like(unique_sess,fill_value=np.nan)
    reconstruction_r2   =  np.full_like(unique_sess,fill_value=np.nan)

    ncond = int(X.shape[0]/unique_sess.size)

    for j,te_run in enumerate(unique_sess):
        # get the test data
        te_filter = session_labels == te_run
        te_X = X[te_filter]
        assert te_X.shape[0] == ncond

        # loop over all possible splits of the remaining (non-test runs) data into training and validation data
        rmruns = [x for x in unique_sess if x!=te_run]
        recons_corr = np.zeros((len(rmruns), ncond)) # 3 runs used as training and validation
        for k,tr_runs in enumerate(itertools.combinations(rmruns, len(rmruns)-1)):
            # get the training data
            tr_filter = np.array([x in tr_runs for x in session_labels])
            tr_X = np.mean(split_data(X=X[tr_filter],groups=session_labels[tr_filter]),axis=0)
            assert tr_X.shape[0] == ncond, f"tr_X shape = {tr_X.shape}, ncond = {ncond}, len(tr_runs) = {len(tr_runs)}"

            # get the validation data
            vld_run = [x for x in rmruns if x not in tr_runs][0]
            vld_filter = session_labels == vld_run
            vld_X = X[vld_filter]
            assert vld_X.shape[0] == ncond

            #perform SVD on training data
            u,s,vh = np.linalg.svd(tr_X,full_matrices=False)

            # loop over components to the correlation between reconstructed training data and validation data
            for ncompo in range(1,ncond):
                
                reconst_train = np.dot(u[:,:ncompo]*s[:ncompo],vh[:ncompo,:])
                recons_corr[k,ncompo-1] = scipy.stats.pearsonr(reconst_train.flatten(),vld_X.flatten()).statistic
                # could also use the following code to do reconstruction:
                # only take the first n components and set the remaining to zero
                # u0,s0,vh0 = np.zeros_like(u),np.zeros_like(s),np.zeros_like(vh)
                # u0[:,:ncompo], s0[:ncompo],vh0[:ncompo,:] = u[:,:ncompo], s[:ncompo],vh[:ncompo,:]
                # reconst_train2 = np.dot(u0*s0,vh0)
                # this should be the same as the curret method, can be tested by running np.array_equal(reconst_train,reconst_train2)

                
        # take the average correlation across the different splits, and get the optimal number of components by finding the maximum correlation
        optdim = int(np.argmax(recons_corr.mean(axis=0))+1) # we add 1 because of zero-indexing

        # get the averaget matrix from the non-test runs
        rm_filter = np.array([x in rmruns for x in session_labels])
        rm_X = np.mean(split_data(X=X[rm_filter],groups=session_labels[rm_filter]),axis=0)
        
        # perform SVD on the average matrix
        u,s,vh = np.linalg.svd(rm_X,full_matrices=False)
        # reconstruct the average matrix using the optimal number of components
        reconst_rm = np.dot(u[:,:optdim]*s[:optdim],vh[:optdim,:])
        reconstruction_corr[j] = scipy.stats.pearsonr(reconst_rm.flatten(),te_X.flatten()).statistic 
        reconstruction_r2[j]   = r2_score(te_X,reconst_rm)
        est_d[j] = optdim

    return est_d, reconstruction_corr, reconstruction_r2, unique_sess