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

#####################################################################################################################################
#                                                        FOR GENERAL PURPOSE                                                        #
#####################################################################################################################################
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

#####################################################################################################################################
#                                                        FOR PIRATE PROJECT                                                         #
#####################################################################################################################################
class ModelRDM:
    """ set up the model rdms based on stimuli properties in pirate exp

    Parameters
    ----------
    stimid : numpy.ndarray
        a 1D numpy array of stimuli ids
    stimgtloc : numpy.ndarray
        a 2D numpy array of stimuli groundtruth locations (x and y), shape = `(nstim,2)`
    stimfeature : numpy.ndarray
        a 2D numpy array of stimuli features (color and feature), shape = `(nstim,2)`
    stimwrttrainloc : numpy.ndarray
        a 2D numpy array of stimuli locations with regard to training location, shape = `(nstim,4)`, the four columns corresponds to: xdist, ydist, xsign, ysign
    stimresploc : numpy.ndarray
        a 2D numpy array of stimuli locations (x and y) based on participants response, shape = `(nstim,2)`, by default None
        number of sessions, by default 1
    n_session : int, optional
        number of sessions, stimuli matrix will be repeated by number of sessions before constructing rdm.
        If multiple sessions are present, model rdms will be separated in to between and within case as well. by default 1
    randomseed: int, optional
        randomseed passed to `self.random` to initiate random generator to create random model rdm, by default 1
    nan_identity: bool, optional
        whether or not to nan out same-same stimuli pairs in the model rdm. by default true.    
    """
    
    def __init__(self,
                 stimid:numpy.ndarray,
                 stimgtloc:numpy.ndarray,
                 stimfeature:numpy.ndarray,
                 stimgroup:numpy.ndarray,
                 stimwrttrainloc:numpy.ndarray,
                 stimresploc:numpy.array=None,
                 n_session:int=1,
                 randomseed:int=1,
                 nan_identity:bool=True):
        
        self.n_session   = n_session
        self.n_stim      = len(stimid)       
        self.stimid      = numpy.tile(stimid,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimloc     = numpy.tile(stimgtloc,(n_session,1)).reshape((self.n_stim*self.n_session,-1)) ## ground-truth location
        self.stimloc_wrttrain = numpy.tile(stimwrttrainloc,(n_session,1)).reshape((self.n_stim*self.n_session,-1)) ## stimuli location with regard to training locations
        self.stimfeature = numpy.tile(stimfeature,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimgroup   = numpy.tile(stimgroup,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimsession = numpy.concatenate([numpy.repeat(j,len(stimid)) for j in range(self.n_session)])
        if stimresploc is None: ## response location
            gen_resprdm = False
        else:
            gen_resprdm = True
            if numpy.shape(stimresploc)[0] == self.stimloc.shape[0]:
                self.stimresploc     = stimresploc
            else:
                self.stimresploc     = numpy.tile(stimresploc,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        
        models = {"gtlocEuclidean":   compute_rdm(self.stimloc,metric="euclidean"),
                  "gtlocCityBlock":   compute_rdm(self.stimloc,metric="cityblock"),
                  "gtloc1dx":         compute_rdm(self.stimloc[:,[0]],metric="euclidean"),
                  "gtloc1dy":         compute_rdm(self.stimloc[:,[1]],metric="euclidean"),
                  "feature2d":        compute_rdm_nomial(self.stimfeature),
                  "feature1dx":       compute_rdm_identity(self.stimfeature[:,0]),
                  "feature1dy":       compute_rdm_identity(self.stimfeature[:,1]),
                  "stimuli":          compute_rdm_identity(self.stimid),
                  "stimuligroup":     compute_rdm_identity(self.stimgroup),
                  "locwrttrain_xydistsign": compute_rdm(self.stimloc_wrttrain,metric="euclidean"),
                  "locwrttrain_xdistsign":  compute_rdm(self.stimloc_wrttrain[:,[0,2]],metric="euclidean"),
                  "locwrttrain_ydistsign":  compute_rdm(self.stimloc_wrttrain[:,[1,3]],metric="euclidean"),
                  "locwrttrain_xdist":      compute_rdm(self.stimloc_wrttrain[:,[0]],metric="euclidean"),
                  "locwrttrain_xsign":      compute_rdm(self.stimloc_wrttrain[:,[2]],metric="euclidean"),
                  "locwrttrain_ydist":      compute_rdm(self.stimloc_wrttrain[:,[1]],metric="euclidean"),
                  "locwrttrain_ysign":      compute_rdm(self.stimloc_wrttrain[:,[3]],metric="euclidean"),
                  "locwrttrain_xysign":     compute_rdm(self.stimloc_wrttrain[:,[2,3]],metric="euclidean"),
                  "locwrttrain_xydist":     compute_rdm(self.stimloc_wrttrain[:,[0,1]],metric="euclidean"),
                  }
        models |= {
                  "shuffledloc2d": self.random(randomseed=randomseed,rdm=models["gtlocEuclidean"],mode="permuterdm"),
                  "randfeature2d": self.random(randomseed=randomseed,mode="randomfeature"),
                  "randmatrix":    self.random(randomseed=randomseed,mode="randommatrix")
                }
        
        if gen_resprdm:
            models |= {"resplocEuclidean":  compute_rdm(self.stimresploc,metric="euclidean"),
                       "resplocCityBlock":  compute_rdm(self.stimresploc,metric="cityblock"),}

        # split into train/test
        if numpy.unique(self.stimgroup).size>1:
            U,V = numpy.meshgrid(self.stimgroup,self.stimgroup)
            WTR = numpy.multiply(1.*(U == 1),1.*(V == 1))
            WTE = numpy.multiply(1.*(U == 0),1.*(V == 0))
            WTR[WTR==0]=numpy.nan
            WTE[WTE==0]=numpy.nan
            if gen_resprdm:
                split_models = ['gtlocEuclidean','gtlocCityBlock','feature2d',
                                'locwrttrain_xydistsign','locwrttrain_xdistsign','locwrttrain_ydistsign','locwrttrain_xdist',
                                'locwrttrain_xsign','locwrttrain_ydist','locwrttrain_ysign','locwrttrain_xysign','locwrttrain_xydist',
                                'resplocEuclidean','resplocCityBlock']
            else:
                split_models = ['gtlocEuclidean','gtlocCityBlock','feature2d',
                                'locwrttrain_xydistsign','locwrttrain_xdistsign','locwrttrain_ydistsign','locwrttrain_xdist',
                                'locwrttrain_xsign','locwrttrain_ydist','locwrttrain_ysign','locwrttrain_xysign','locwrttrain_xydist']
            for k in split_models:
                wtr_n = 'trainstimpairs_' + k
                rdmwtr = numpy.multiply(models[k],WTR)
                wte_n = 'teststimpairs_' + k
                rdmwte = numpy.multiply(models[k],WTE)
                models |= {wtr_n:rdmwtr,wte_n:rdmwte}

        # split into sessions
        if n_session>1:
            BS = compute_rdm_identity(self.stimsession) # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=numpy.nan
            WS[WS==0]=numpy.nan
            tmp = list(models.items()) ## this is so that we don't change models
            for k,v in tmp:
                ws_n  = 'within_'+k
                rdmws = numpy.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = numpy.multiply(v,BS)
                models |= {ws_n:rdmws,bs_n:rdmbs}
            models["session"] = compute_rdm_identity(self.stimsession)
        for k,v in models.items():
            if nan_identity:
                from copy import deepcopy
                nan_matrix = deepcopy(models["stimuli"])
                nan_matrix[numpy.where(nan_matrix==0)] = numpy.nan
                models[k] = numpy.multiply(v,nan_matrix)
        self.models = models


    def __str__(self):
        return 'The following model rdms are created:\n' + ',\n'.join(
            self.models.keys()
        )
    
    def random(self,randomseed:int=1,rdm:numpy.ndarray=None,mode:str="randommatrix") -> numpy.ndarray:
        """generate random model RDM based on sample size

        Parameters
        ----------
        randomseed: int
            a random seed used for the random state instance that are used for generating random matrix/feature/permutation
        rdm : numpy.ndarray
            a 2D numpy array of a model rdm that is used for shuffling to generate random RDM
        mode : str
            a string specifying the kind of random rdm to be generated. Must be one of: randomfeature, permuterdm,randommatrix

        Returns
        -------
        numpy.ndarray
            2D numpy array of a random model rdm
        """
        
        prng = numpy.random.RandomState(randomseed)
        if rdm is not None:
            assert isinstance(rdm,numpy.ndarray), "rdm must be 2D square array of size = (nsample,nsample)"
            assert rdm.shape == (self.n_stim*self.n_session,self.n_stim*self.n_session), "rdm must be 2D square array of size = (nsample,nsample)"
            mode = "permuterdm"

        if mode == "permuterdm":
            return prng.permutation(rdm.flatten()).reshape(rdm.shape)
        elif mode == "randomfeature":
            rand_features = prng.random(size=self.stimfeature.shape)
            return compute_rdm(rand_features,"euclidean")
        elif mode == "randommatrix":
            return prng.random((self.n_stim*self.n_session,self.n_stim*self.n_session))
        else:
            raise ValueError("invalid random mode, must be one of: randomfeature, permuterdm, randommatrix")    

    def visualize(self,modelname:str or list="all",tri:int=0,annot:bool=False)->matplotlib.figure:
        """plot model rdms using seaborn heatmap

        Parameters
        ----------
        modelname : str or list of strings, optional
            the name of model rdm to be plotted, by default "all"
        tri: int, optional
            show lower(0), upper(1) or the whole matrix. lower and upper will exclude diagonal elements by default 0
        annot: bool, optional
            show the value in the heatmap, by default False

        Returns
        -------
        matplotlib.figure
            the plotted figure
        """
        if isinstance(modelname,list):
            if numpy.all([m in self.models.keys() for m in modelname]):
                plot_models = modelname
            else:
                plot_models = list(self.models.keys())
        elif isinstance(modelname,str):
            if modelname in self.models.keys():
                plot_models = [modelname]
            elif modelname == "all":
                plot_models = list(self.models.keys())
            else:
                print("invalid model name, plotting all models")
                plot_models = list(self.models.keys())

        n_model = len(plot_models)
        n_row = int(numpy.sqrt(n_model))
        n_col = int(numpy.ceil(n_model/n_row))
        fig,axes = plt.subplots(n_row,n_col,figsize = (5*n_col, 5*n_row))
        for j,k in enumerate(plot_models):
            v = numpy.full_like(self.models[k],numpy.nan)
            if tri==0:
                _,idx = lower_tri(v)
            elif tri==1:
                _,idx = upper_tri(v)
            elif tri==2:
                idx = numpy.where(numpy.isnan(v))
            v[idx] = self.models[k][idx]

            if n_model>1:
                sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85},annot=annot)
                axes.flatten()[j].set_title(k)
            else:
                sns.heatmap(v,ax=axes,square=True,cbar_kws={"shrink":0.85},annot=annot)
                axes.set_title(k)
        if n_row*n_col>1:
            for k in numpy.arange(numpy.size(axes.flatten())-1-j)+1:
                fig.delaxes(axes.flatten()[j+k])
        return fig
    
    def rdm_to_df(self,modelnames:list or str,rdms:list or numpy.ndarray=None) -> pandas.DataFrame:
        """put the lower triangular part (excluding the diagonal) fo a square rdm matrix into pandas dataframe indexed by stimuli id, run(session), and group

        Parameters
        ----------
        modelnames : list or str
            a model name or a list of model names. If rdms is not specified, the model names must be one of the keys of `self.models`
        rdms : listornumpy.ndarray, optional
            the external rdms (that are generated for the same stimuli pairs as in `self.models` but based on different feature vectors) to be transformed, by default None

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(modelnames,list):
            if isinstance(modelnames,str):
                modelnames = [modelnames]
            else:
                raise ValueError("modelnames must be a model name or a list of model names")
        
        if rdms is None:
            assert numpy.all([x in self.models.keys() for x in modelnames]), "Invalid model names!"
            rdms = [self.models[m] for m in modelnames]
        else:
            if not isinstance(rdms,list):
                if isinstance(rdms,numpy.ndarray):
                    rdms = [rdms]
                else:
                    raise ValueError("rdms must be a 2d numpy array of rdm or a list of 2d numpy array rdms")            
            assert len(rdms)==len(modelnames), "number of rdms do not match number of names"

        modeldf = []
        for m,rdm in zip(modelnames,rdms):
            lt,idx = lower_tri(rdm)
            c = numpy.repeat(numpy.arange(0, rdm.shape[0]), rdm.shape[0]).reshape(rdm.shape)[idx]
            i = numpy.tile(numpy.arange(0, rdm.shape[0]),  rdm.shape[0]).reshape(rdm.shape)[idx]
            df = pandas.DataFrame({'stimidA':numpy.squeeze(self.stimid[c]), 'stimidB': numpy.squeeze(self.stimid[i]), 
                                   'groupA': numpy.squeeze(self.stimgroup[c]),  'groupB': numpy.squeeze(self.stimgroup[i]), 
                                   'runA': numpy.squeeze(self.stimsession[c]),  'runB': numpy.squeeze(self.stimsession[i]), 
                                    m: lt}
                                 ).set_index(['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB'])
            modeldf.append(df)
        modeldf = modeldf[0].join(modeldf[1:])
        return modeldf
        