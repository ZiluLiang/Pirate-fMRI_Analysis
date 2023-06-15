import scipy
from scipy.spatial.distance import pdist, squareform
import numpy
import pandas
import os
import seaborn as sns
import matplotlib.pyplot as plt

def lower_tri(rdm):
    lower_tril_idx = numpy.tril_indices(rdm.shape[0], k = -1)
    rdm_tril = rdm[lower_tril_idx]
    return rdm_tril

def upper_tri(rdm):
    upper_triu_idx = numpy.triu_indices(rdm.shape[0], k = 1)
    rdm_triu = rdm[upper_triu_idx]
    return rdm_triu       

def compute_R2(yhat,y,nparam) -> tuple:
    # retrieved from 
    # https://stackoverflow.com/questions/42033720/python-sklearn-multiple-linear-regression-display-r-squared
    # compute with formulas from the theory
    SS_Residual = sum((y-yhat)**2)       
    SS_Total = sum((y-numpy.mean(y))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-nparam-1)
    return r_squared, adjusted_r_squared

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
    if not pattern_matrix.ndim == 2:
        raise Exception("pattern matrix must be 2D")
    else:
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
        raise Exception("invalid input type")
    
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

class ModelRDM:
    def __init__(self,stimid,stimloc,stimfeature,n_session:int=1,split_sess:bool=True):
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
        
    def session(self):
        S = numpy.zeros((self.n_stim,self.n_stim)) # matrix for same session
        D = numpy.ones((self.n_stim,self.n_stim)) # matrix for different session
        
        r = []
        for j in range(self.n_session):
            c = [D for _ in range(self.n_session)]
            c[j] = S
            r.append(c)
        modelrdm = numpy.block(r)
        return modelrdm
    
    def euclidean2d(self):
        modelrdm = compute_rdm(self.stimloc,metric="euclidean")
        return modelrdm

    def euclidean1d(self,dim):
        X,Y = numpy.meshgrid(self.stimloc[:,dim],self.stimloc[:,dim])
        modelrdm = abs(X-Y)
        return modelrdm

    def feature2d(self):
        RDM_attrx = self.feature1d(0)
        RDM_attry = self.feature1d(1)
        #modelrdm  = numpy.sqrt(RDM_attrx+RDM_attry)
        modelrdm  = RDM_attrx+RDM_attry
        return modelrdm
    
    def feature1d(self,dim):
        X,Y = numpy.meshgrid(self.stimfeature[:,dim],self.stimfeature[:,dim])
        modelrdm = 1-abs(X==Y)
        return modelrdm

    def stimuli(self):
        RDM_attrx = self.feature1d(0)
        RDM_attry = self.feature1d(1)
        modelrdm  = 1 * (RDM_attrx+RDM_attry)==0
        return modelrdm
    
    def visualize(self,modelname:str="all"):
        if modelname in self.models.keys():
            sns.heatmap(self.models[modelname])
        elif modelname == "all":
            n_model = len(self.models.keys())
            n_row = int(numpy.sqrt(n_model))
            n_col = int(numpy.ceil(n_model/n_row))
            fig,axes = plt.subplots(n_row,n_col,figsize = (5*n_col, 5*n_row))
            for j,(k,v) in enumerate(self.models.items()):
                sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
                axes.flatten()[j].set_title(k)
            for k in numpy.arange(numpy.size(axes.flatten())-1-j)+1:
                fig.delaxes(axes.flatten()[j+k])
        else:
            print("invalid model name")
        