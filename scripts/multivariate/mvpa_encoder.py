import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import os
import nibabel as nib
import nibabel.processing
from nilearn.masking import apply_mask
import sklearn
    
class LinearEncoder:
    def __init__(self,patternmatrix,featurematrix,group) -> None:
        group = np.array(group)
        X = featurematrix
        Y = patternmatrix

        self.trainX = self.standardize(X[np.where(group==0)])
        self.trainY = self.standardize(Y[np.where(group==0)])
        self.testX  = self.standardize(X[np.where(group==1)])
        self.testY  = self.standardize(Y[np.where(group==1)])

        self.Nf = np.shape(X)[1] # number of features
        if Y.ndim > 1:
            self.Nr = np.shape(Y)[1] # number of voxels/regions
        else:
            self.Nr = 1
    
    def fit(self):
        self.encoder = LinearRegression().fit(
            self.trainX,
            self.trainY,
            )
        self.W  = self.encoder.coef_
        return self
    
    def validate(self):
        self.pred_testY = self.encoder.predict(self.testX)
        if self.Nr>1:
            self.score_profile = [scipy.stats.spearmanr(self.pred_testY[:,r],self.testY[:,r]).correlation for r in range(self.Nr)]
            self.score_stim    = [scipy.stats.spearmanr(self.pred_testY[s,:],self.testY[s,:]).correlation for s in range(self.testX.shape[0])]        
        else:
            self.score_profile = scipy.stats.spearmanr(self.pred_testY,self.testY).correlation
            self.score_stim    = np.nan
        self.score_sklearn = self.encoder.score(self.testX, self.testY)
        return self
        
    def standardize(self,X):
        # standardize
        X_standardized = (X - np.mean(X))/np.std(X)
        return X_standardized
