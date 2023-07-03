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

class ModelRDM:
    """ set up the model rdms

    Parameters
    ----------
    stimid : numpy.ndarray
        a 1D numpy array of stimuli ids
    stimloc : numpy.ndarray
        a 2D numpy array of stimuli locations (x and y)
    stimfeature : numpy.ndarray
        a 2D numpy array of stimuli features (color and feature)
    n_session : int, optional
        number of sessions, by default 1
    split_sess : bool, optional
        split the rdm into within-session and between-session or not, by default True
    """
    
    def __init__(self,
                 stimid:numpy.ndarray,
                 stimloc:numpy.ndarray,
                 stimfeature:numpy.ndarray,
                 stimgroup:numpy.ndarray,
                 n_session:int=1,
                 randomseed:int=1):
        self.n_session   = n_session
        self.n_stim      = len(stimid)       
        self.stimid      = numpy.tile(stimid,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimloc     = numpy.tile(stimloc,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimfeature = numpy.tile(stimfeature,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimgroup   = numpy.tile(stimgroup,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimsession = numpy.concatenate([numpy.repeat(j,len(stimid)) for j in range(self.n_session)])

        models = {"loc2d":self.euclidean2d(),
                  "loc1dx":self.euclidean1d(0),
                  "loc1dy":self.euclidean1d(1),
                  "feature2d":self.feature2d(),
                  "feature1dx":self.feature1d(0),
                  "feature1dy":self.feature1d(1),
                  "stimuli":self.identity(),
                  "stimuligroup":self.identity(self.stimgroup),
                  "shuffledloc2d":self.random(randomseed=randomseed,rdm=self.euclidean2d(),mode="permuterdm"),
                  "randfeature2d":self.random(randomseed=randomseed,mode="randomfeature"),
                  "randmatrix":self.random(randomseed=randomseed,mode="randommatrix")
                }

        # split into sessions
        if n_session>1:
            BS = self.session() # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=numpy.nan
            WS[WS==0]=numpy.nan
            tmp = list(models.items())
            for k,v in tmp:
                ws_n  = 'within_'+k
                rdmws = numpy.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = numpy.multiply(v,BS)
                models |= {ws_n:rdmws,bs_n:rdmbs}
            models["session"] = self.session()
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

    def session(self)->numpy.ndarray:
        """calculate model rdm based on session, if the pair is in the same session, distance will be 0, otherwise will be 1
        
        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        S = numpy.zeros((self.n_stim,self.n_stim)) # matrix for same session
        D = numpy.ones((self.n_stim,self.n_stim)) # matrix for different session
        
        r = []
        for j in range(self.n_session):
            c = [D for _ in range(self.n_session)]
            c[j] = S
            r.append(c)
        modelrdm = numpy.block(r)
        return modelrdm
    
    def euclidean2d(self)->numpy.ndarray:
        """calculate model rdm based on 2d euclidean distance.
        
        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        modelrdm = compute_rdm(self.stimloc,metric="euclidean")
        return modelrdm

    def euclidean1d(self,dim)->numpy.ndarray:
        """calculate model rdm based on 1d euclidean distance.

        Parameters
        ----------
        dim : int
        which axis (0:x, 1:y) should be used to calculated rdm.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        X,Y = numpy.meshgrid(self.stimloc[:,dim],self.stimloc[:,dim])
        modelrdm = abs(X-Y)
        return modelrdm

    def feature2d(self)->numpy.ndarray:
        """calculate model rdm based on both stimuli features, if the pair:
            shares 0 feature  - sqrt(2);
            shares 1 feature  - 1;
            shares 2 features - 0;

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        RDM_attrx = self.feature1d(0)
        RDM_attry = self.feature1d(1)
        modelrdm  = numpy.sqrt(RDM_attrx+RDM_attry)
        return modelrdm
    
    def feature1d(self,dim:int)->numpy.ndarray:
        """calculate model rdm based on 1 stimuli feature, if the pair has the same feature, distance will be zero, otherwise will be one.

        Parameters
        ----------
        dim : int
        which feature should be used to calculated rdm.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        X,Y = numpy.meshgrid(self.stimfeature[:,dim],self.stimfeature[:,dim])
        modelrdm = 1. - abs(X==Y) # if same feature, distance=0
        return modelrdm
    

    def identity(self,identity_arr=None)->numpy.ndarray:
        """calculate model rdm based on stimuli identity, if the pair is the same stimuli, distance will be zero, otherwise will be one.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        if identity_arr is None:
            X,Y = numpy.meshgrid(self.stimid,self.stimid)
        else:
            X,Y = numpy.meshgrid(identity_arr,identity_arr)
        modelrdm = 1. - abs(X==Y)# if same stimuli, distance=0
        return modelrdm
    
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
        