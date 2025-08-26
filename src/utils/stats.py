import scipy
import numpy as np


def ttest1samp_equiv_permutation(y,n_perm=10000,return_ponly = True,**kwargs):
    res = scipy.stats.permutation_test((y,),
                                       statistic=np.mean,
                                       n_resamples=n_perm,
                                       permutation_type="samples",
                                       random_state  = 0, # for reproducibility
                                       **kwargs)
    if return_ponly:
        return res.pvalue
    else:
        return res

def ttestpairedsamp_equiv_permutation(y1,y2,n_perm=10000,return_ponly = True,**kwargs):
    res = scipy.stats.permutation_test((y1,y2),
                                       statistic=lambda y1,y2: np.mean(y1-y2),
                                       n_resamples=n_perm,permutation_type="samples",
                                       random_state = 0, # for reproducibility
                                       **kwargs)
    if return_ponly:
        return res.pvalue
    else:
        return res
    
def ttestindsamp_equiv_permutation(y1,y2,n_perm=10000,return_ponly = True,**kwargs):
    res = scipy.stats.permutation_test((y1,y2),
                                       statistic=lambda y1,y2: np.mean(y1)-np.mean(y2),
                                       n_resamples=n_perm,permutation_type="independent",
                                       random_state  = 0, # for reproducibility
                                       **kwargs)
    if return_ponly:
        return res.pvalue
    else:
        return res

def correlation_equiv_permutation(y1,y2,corrfun,n_perm=10000,return_ponly = True,**kwargs):
    res = scipy.stats.permutation_test((y1,y2),
                                       statistic=lambda y1,y2: corrfun(y1,y2),
                                       n_resamples=n_perm,permutation_type="pairings",
                                       random_state  = 0, # for reproducibility
                                       **kwargs)
    if return_ponly:
        return res.pvalue
    else:
        return res
    
def compute_se(x1:np.ndarray,x2=None):
    """Compute the standard error of the sample means (SEM) for one or two samples.

    Parameters
    ----------
    x1 : np.ndarray
        data for the first sample
    x2 : np.ndarray, optional
        data for the second sample, by default None

    Returns
    -------
    SEM : float
        Standard error of the mean for one or two samples.
    """
    if x2 is None:
        return np.std(x1, ddof=1) / np.sqrt(len(x1))
    else:
        return np.sqrt(np.var(x1, ddof=1) / len(x1) + np.var(x2, ddof=1) / len(x2))
