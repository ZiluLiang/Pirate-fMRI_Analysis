import scipy
from scipy.spatial.distance import pdist, squareform
import numpy
import pandas
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def standardize_2D(X:numpy.ndarray,dim:int=2)->numpy.ndarray:
    """ standardize a 2D numpy array by using ZX = (X - mean)/std

    Parameters
    ----------
    X : numpy.ndarray
        the 1D or 2D numpy array that needs to be normalized
    dim : int, optional
        if 0, will perfrom standardization independently for each row
        if 1, will perform standardization independently for each column
        if 2, will perform standardization on the whole matrix
        by default 2

    Returns
    -------
    numpy.ndarray
        standardized 2D array X
    """
    assert isinstance(X,numpy.ndarray), "X must be numpy array"
    assert X.ndim <= 2, "X must be 1D or 2D"
    
    if X.ndim == 1:
        dim = 2
    if dim == 2:
        ZX = (X - numpy.mean(X)) / numpy.std(X)
    else:
        if dim == 1:
            X = X.T
        row_means = X.mean(axis=1)
        row_stds  = X.std(axis=1)
        ZX = (X - row_means[:, numpy.newaxis]) / row_stds[:, numpy.newaxis]
        if dim == 1:
            ZX = ZX.T
    return ZX


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
    if not isinstance(dirs, list):
        if isinstance(dirs,str):
            dirs = [dirs]
    else:
        raise AssertionError("dirs must be a list of directory strings or a directory string")
    
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

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
    def __init__(self,stimid:numpy.ndarray,stimloc:numpy.ndarray,stimfeature:numpy.ndarray,n_session:int=1,split_sess:bool=True):
        self.n_session   = n_session
        self.n_stim      = len(stimid)
        self.stimid      = numpy.tile(stimid,(n_session,1))
        self.stimloc     = numpy.tile(stimloc,(n_session,1))
        self.stimfeature = numpy.tile(stimfeature,(n_session,1))
        models = {"session":self.session(),
                  "loc2d":self.euclidean2d(),
                  "loc1dx":self.euclidean1d(0),
                  "loc1dy":self.euclidean1d(1),
                  "feature2d":self.feature2d(),
                  "feature1dx":self.feature1d(0),
                  "feature1dy":self.feature1d(1),
                  "stimuli":self.stimuli(),
                }
        if not split_sess:
            self.models = models
        else:
            BS = self.session() # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=numpy.nan
            WS[WS==0]=numpy.nan
            models_split = dict({"session":models.pop("session")})
            for k,v in models.items():
                ws_n  = 'within_'+k
                rdmws = numpy.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = numpy.multiply(v,BS)
                models_split.update({ws_n:rdmws,bs_n:rdmbs})
            self.models = models_split

    def __str__(self):
        summary = 'The following model rdms are created:\n' + ',\n'.join(self.models.keys())
        return summary

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
        modelrdm = 1-abs(X==Y) # if same feature, distance=0
        return modelrdm

    def stimuli(self)->numpy.ndarray:
        """calculate model rdm based on stimuli identity, if the pair is the same stimuli, distance will be zero, otherwise will be one.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        X,Y = numpy.meshgrid(self.stimid,self.stimid)
        modelrdm = 1-abs(X==Y)# if same stimuli, distance=0
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
        