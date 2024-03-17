"""
This module contains helper function for general purpose

Zilu Liang @HIPlab Oxford
2023
"""
import scipy
from scipy.spatial.distance import pdist, squareform
import numpy
import nibabel
import nibabel.processing
import pandas
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas

def scale_feature(X:numpy.ndarray,s_dir:int=2,standardize:bool=True) -> numpy.ndarray:
    """ standardize or center a 1D or 2D numpy array by using ZX = (X - mean)/std

    Parameters
    ----------
    X : numpy.ndarray
        the 1D or 2D numpy array that needs to be normalized
    s_dir : int, optional
        the direction along which to perform standardization
        if 0, will perfrom standardization independently for each row  \n
        if 1, will perform standardization independently for each column  \n      
        if 2, will perform standardization on the whole matrix  \n
        by default 2

    Returns
    -------
    numpy.ndarray
        standardized 1D or 2D array ZX
    """
    assert isinstance(X,numpy.ndarray), "X must be numpy array"
    assert X.ndim <= 2, "X must be 1D or 2D"

    if X.ndim == 1:
        s_dir = 2

    if s_dir == 0:
        ZX = _rowwise_standardize(X,standardize)
    elif s_dir == 1:
        X = X.T
        ZX = _rowwise_standardize(X,standardize)
        ZX = ZX.T
    elif s_dir == 2:
        denom = numpy.std(X) if standardize else 1
        ZX = (X - numpy.mean(X)) / denom
    return ZX


def _rowwise_standardize(X,standardize:bool):
    row_means = X.mean(axis=1)
    row_stds  = X.std(axis=1)
    denom = row_stds[:, numpy.newaxis] if standardize else 1
    return (X - row_means[:, numpy.newaxis]) / denom


def lower_tri(rdm:numpy.ndarray) -> tuple:
    """return the lower triangular part of the RDM, excluding the diagonal elements

    Parameters
    ----------
    rdm : numpy.ndarray
        a 2D numpy array. the representation dissimilarity matrix. 

    Returns
    -------
    tuple: (rdm_tril,lower_tril_idx)
        rdm_tril: the lower triangular part of the RDM, excluding the diagonal elements
        lower_tril_idx: the index of the lower triangular part of the RDM, excluding the diagonal elements
    """
    assert isinstance(rdm,numpy.ndarray), "rdm must be 2D numpy array"
    assert rdm.ndim == 2, "rdm must be 2D numpy array"
    
    lower_tril_idx = numpy.tril_indices(rdm.shape[0], k = -1)
    rdm_tril = rdm[lower_tril_idx]
    return rdm_tril,lower_tril_idx

def upper_tri(rdm:numpy.ndarray) -> tuple:
    """return the upper triangular part of the RDM, excluding the diagonal elements

    Parameters
    ----------
    rdm : numpy.ndarray
        a 2D numpy array. the representation dissimilarity matrix. 

    Returns
    -------
    tuple: (rdm_tril,upper_tril_idx)
        rdm_tril: a 1D numpy array. the upper triangular part of the RDM, excluding the diagonal elements
        upper_tril_idx: a 1D numpy array. the index of the upper triangular part of the RDM, excluding the diagonal elements
    """

    assert isinstance(rdm,numpy.ndarray), "rdm must be 2D numpy array"
    assert rdm.ndim == 2, "rdm must be 2D numpy array"

    upper_triu_idx = numpy.triu_indices(rdm.shape[0], k = 1)
    rdm_triu = rdm[upper_triu_idx]
    return rdm_triu,upper_triu_idx

def compute_R2(y_pred:numpy.ndarray, y_true:numpy.ndarray, nparam: int) -> tuple:
    """compute the coefficient of determination (R-square or adjusted R-square) of a model based on prediction and true value
    based on formula in https://en.wikipedia.org/wiki/Coefficient_of_determination

    Parameters
    ----------
    y_pred : numpy.ndarray
        1D numpy array of predicted y values
    y_true : numpy.ndarray
        1D numpy array of true (observed) y values
    nparam : int
        number of parameters in the model

    Returns
    -------
    tuple
        a tuple of (r-squared, adjusted r-squared)
    """
    SS_Residual = numpy.sum((y_true-y_pred)**2)       
    SS_Total = numpy.sum((y_true-numpy.mean(y_true))**2)     
    R_squared = 1 - SS_Residual/SS_Total
    n_sample = len(y_true)
    adjusted_R_squared = 1 - (1-R_squared)*(n_sample-1)/(n_sample-nparam)
    return R_squared, adjusted_R_squared

def compute_rdm(pattern_matrix:numpy.ndarray,metric:str) -> numpy.ndarray:
    """compute the dissimilarity matrix of a nsample x nfeature matrix

    Parameters
    ----------
    pattern_matrix : numpy.ndarray
        a 2D numpy array of size: nsample x nfeature
    metric : str
        dissimilarity/distance metric passed to `scipy.spatial.distance.pdist`

    Returns
    -------
    numpy.ndarray
        a nsample x nsample dissimliarity matrix

    Raises
    ------
    Exception
        pattern matrix must be 2D
    """
    assert isinstance(pattern_matrix,numpy.ndarray), "pattern_matrix must be 2D numpy array"
    assert pattern_matrix.ndim == 2, "pattern_matrix must be 2D numpy array"
    
    X = pattern_matrix
    na_filters = numpy.all([~numpy.isnan(X[j,:]) for j in range(numpy.shape(X)[0])],0)
    X_drop_na = X[:,na_filters]
    rdm = squareform(pdist(X_drop_na, metric=metric))    
    return rdm

def compute_rdm_identity(identity_arr:numpy.ndarray) -> numpy.ndarray:
    """calculate model rdm based on stimuli identity, if the pair have the same value, distance will be zero, otherwise will be one.

    Parameters
    ----------
    identity_arr: numpy.ndarray
        a 1D numpy array of size: (nsample,)

    Returns
    -------
    numpy.ndarray
        2D numpy array of model rdm
    """
    assert isinstance(identity_arr,numpy.ndarray), "identity_arr must be 1D numpy array"
    identity_arr = numpy.squeeze(identity_arr)
    assert identity_arr.ndim == 1, "identity_arr must be 1D numpy array"
    X,Y = numpy.meshgrid(identity_arr,identity_arr)
    return 1. - abs(X==Y)# if same, distance=0

def compute_rdm_nomial(pattern_matrix:numpy.ndarray) -> numpy.ndarray:
    """compute the dissimilarity matrix of a nsample x nfeature matrix where feature values are nomial.
    Features are assumed to be orthogonal so the distance will be Euclidean distance assuming features are one-hot encoded.

    Parameters
    ----------
    pattern_matrix : numpy.ndarray
        a 2D numpy array of size: nsample x nfeature
    
    Returns
    -------
    numpy.ndarray
        a nsample x nsample dissimliarity matrix

    Raises
    ------
    Exception
        pattern matrix must be 2D
    """
    assert isinstance(pattern_matrix,numpy.ndarray), "pattern_matrix must be 2D numpy array"
    assert pattern_matrix.ndim == 2, "pattern_matrix must be 2D numpy array"
    
    X = pattern_matrix
    na_filters = numpy.all([~numpy.isnan(X[j,:]) for j in range(X.shape[0])],0)
    X_drop_na = X[:,na_filters]
    feature_rdms = [compute_rdm_identity(X_drop_na[:,k]) for k in range(X_drop_na.shape[1])]
    rdm = numpy.sqrt(numpy.sum(feature_rdms,axis=0))
    return rdm

def checkdir(dirs:list or str):
    """check if directories exist, if not, generate directories

    Parameters
    ----------
    dirs : list or str
        a list of directory strings or a directory string

    Raises
    ------
    Exception
        input must be list or string
    """
    if isinstance(dirs, list):
        raise AssertionError("dirs must be a list of directory strings or a directory string")

    if isinstance(dirs,str):
        dirs = [dirs]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def load_maskimg(mask_img: str or nibabel.Nifti1Image or numpy.ndarray, ref_maskimg: str or nibabel.Nifti1Image) -> nibabel.Nifti1Image:
    """load a mask image and reslice to the same resolution as the reference image

    Parameters
    ----------
    mask_img : str or nibabel.Nifti1Image
        the loaded mask image, or directory to the mask image, or ndarray containing the mask image data
    ref_maskimg : str or nibabel.Nifti1Image
        the loaded reference image or directory to the reference image

    Returns
    -------
    nibabel.Nifti1Image
        loaded and resampled mask image
    """
    if ref_maskimg(mask_img, str):
        ref_maskimg = nibabel.load(ref_maskimg)
    elif isinstance(ref_maskimg,nibabel.Nifti1Image):
        ref_maskimg = ref_maskimg
    else:
        raise AssertionError("ref_maskimg must be the path to nii image or nibabel loaded nii image")

    if isinstance(mask_img, str):
        mask_img = nibabel.load(mask_img)
    elif isinstance(mask_img, numpy.ndarray):
        reshapeimg = numpy.reshape(mask_img,ref_maskimg).astype(numpy.int8)
        mask_img = nibabel.Nifti1Image(reshapeimg, ref_maskimg.affine, ref_maskimg.header)
    elif isinstance(ref_maskimg,nibabel.Nifti1Image):
        mask_img = mask_img
    else:
        raise AssertionError("mask_img must be the path to nii image or nibabel loaded nii image, or a ndarry of mask image data")
    return nibabel.processing.resample_from_to(mask_img, ref_maskimg)
