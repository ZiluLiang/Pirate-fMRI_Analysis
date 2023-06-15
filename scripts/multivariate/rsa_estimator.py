import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

class PatternCorrelation:
    def __init__(self,neuralrdm,modelrdm) -> None:
        X = self.lower_tri(neuralrdm)
        Y = self.lower_tri(modelrdm)
        na_filters = np.logical_and(~np.isnan(X),~np.isnan(Y))
        self.X = self.standardize(X[na_filters])
        self.Y = self.standardize(Y[na_filters])
    
    def fit(self):                
        self.result   = scipy.stats.spearmanr(self.X,self.Y).correlation
        return self
    
    def lower_tri(self,rdm):
        lower_tril_idx = np.tril_indices(rdm.shape[0], k = -1)
        rdm_tril = rdm[lower_tril_idx]
        return rdm_tril
        
    def standardize(self,X):
        # standardize
        X_standardized = (X - np.mean(X))/np.std(X)
        return X_standardized
    

class MultipleRDMRegression:
    def __init__(self,neuralrdm,modelrdms) -> None:
        
        Y = self.lower_tri(neuralrdm)

        self.n_reg = len(modelrdms) # number of model rdms
        X = np.empty((len(Y),self.n_reg))
        for j,m in enumerate(modelrdms):
            X[:,j] = self.lower_tri(m)

        xna_filters = np.all([~np.isnan(X[:,j]) for j in range(np.shape(X)[1])],0)
        na_filters = np.logical_and(~np.isnan(Y),xna_filters)

        sX = []
        X_dropNA = X[na_filters,:]
        #standardize each column
        for j in range(self.n_reg):
            sX.append(self.standardize(X_dropNA[:,j]))        
        self.X = np.array(sX).T

        self.Y = self.standardize(Y[na_filters])
        self.n_sample = len(self.Y)# number of elements in the lower triagular of the rdm
    
    def fit(self):
        reg = LinearRegression().fit(self.X,self.Y)
        self.result = reg.coef_
        return self
    
    def lower_tri(self,rdm):
        lower_tril_idx = np.tril_indices(rdm.shape[0], k = -1)
        rdm_tril = rdm[lower_tril_idx]
        return rdm_tril
        
    def standardize(self,X):
        # standardize
        X_standardized = (X - np.mean(X))/np.std(X)
        return X_standardized
