import scipy
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

def compute_rdm(pattern_matrix,metric):
    if pattern_matrix.ndim<2:
        raise Exception("pattern matrix must have at least two features")
    else:
        X = pattern_matrix
        na_filters = np.all([~np.isnan(X[j,:]) for j in range(np.shape(X)[0])],0)
        X_drop_na = X[:,na_filters]
        rdm = squareform(pdist(X_drop_na, metric=metric))    
    return rdm

import os
def checkdir(dirs):
    if not isinstance(dirs, list):
        if isinstance(dirs,str):
            dirs = [dirs]
    else:
        raise Exception("invalid input type")
    
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

class ModelRDM:
    def __init__(self,stimid,stimloc,stimfeature,n_session:int=1,cv_sess:bool=True):
        self.n_session   = n_session
        self.n_stim      = len(stimid)
        self.stimid      = np.tile(stimid,(n_session,1))
        self.stimloc     = np.tile(stimloc,(n_session,1))
        self.stimfeature = np.tile(stimfeature,(n_session,1))
        models = {"session":self.session(),
                  "loc2d":self.euclidean2d(),
                  "loc1dx":self.euclidean1d(0),
                  "loc1dy":self.euclidean1d(1),
                  "feature2d":self.feature2d(),
                  "feature1dx":self.feature1d(0),
                  "feature1dy":self.feature1d(1),
                }
        if not cv_sess:
            self.models = models
        else:
            BS = self.session() # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=np.nan
            WS[WS==0]=np.nan
            models_split = dict({"session":models.pop("session")})
            for k,v in models.items():
                ws_n  = 'within_'+k
                rdmws = np.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = np.multiply(v,BS)
                models_split.update({ws_n:rdmws,bs_n:rdmbs})
            self.models = models_split
        
    def session(self):
        S = np.zeros((self.n_stim,self.n_stim)) # matrix for same session
        D = np.ones((self.n_stim,self.n_stim)) # matrix for different session
        
        r = []
        for j in range(self.n_session):
            c = [D for _ in range(self.n_session)]
            c[j] = S
            r.append(c)
        modelrdm = np.block(r)
        return modelrdm
    
    def euclidean2d(self):
        modelrdm = compute_rdm(self.stimloc,metric="euclidean")
        return modelrdm

    def euclidean1d(self,dim):
        X,Y = np.meshgrid(self.stimloc[:,dim],self.stimloc[:,dim])
        modelrdm = abs(X-Y)
        return modelrdm

    def feature2d(self):
        RDM_attrx = self.feature1d(0)
        RDM_attry = self.feature1d(1)
        #modelrdm  = np.sqrt(RDM_attrx+RDM_attry)
        modelrdm  = RDM_attrx+RDM_attry
        return modelrdm
    
    def feature1d(self,dim):
        X,Y = np.meshgrid(self.stimfeature[:,dim],self.stimfeature[:,dim])
        modelrdm = 1-abs(X==Y)
        return modelrdm

    def visualize(self,modelname:str="all"):
        if modelname in self.models.keys():
            sns.heatmap(self.models[modelname])
        elif modelname == "all":
            n_model = len(self.models.keys())
            n_row = int(np.sqrt(n_model))
            n_col = int(np.ceil(n_model/n_row))
            fig,axes = plt.subplots(n_row,n_col,figsize = (5*n_col, 5*n_row))
            for j,(k,v) in enumerate(self.models.items()):
                sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
                axes.flatten()[j].set_title(k)
            for k in np.arange(np.size(axes.flatten())-1-j)+1:
                fig.delaxes(axes.flatten()[j+k])
        else:
            print("invalid model name")
        