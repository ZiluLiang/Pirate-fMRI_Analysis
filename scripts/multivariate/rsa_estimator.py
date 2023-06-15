import numpy
import scipy
from sklearn.linear_model import LinearRegression
from multivariate.helper import lower_tri, upper_tri, standardize, compute_R2
import matplotlib.pyplot as plt
import seaborn as sns

class PatternCorrelation:
    """calculate the correlation between neural rdm and model rdm

    Parameters
    ----------
    neuralrdm : numpy.ndarray
        a 2D numpy array. the neural representation dissimilarity matrix. 
    modelrdm : numpy.ndarray
        a 2D numpy array. the model representation dissimilarity matrix. 
    type : str, optional
        type of correlation measure, by default "spearman".
        must be one of: "spearman", "pearson", "kendall", "linreg"
    """
    def __init__(self,neuralrdm:numpy.ndarray,modelrdm:numpy.ndarray,type:str="spearman") -> None:
        self.rdm_shape = neuralrdm.shape
        
        X,_ = lower_tri(neuralrdm)
        Y,_ = lower_tri(modelrdm)
        na_filters = numpy.logical_and(~numpy.isnan(X),~numpy.isnan(Y))
        self.X = X[na_filters]
        self.Y = Y[na_filters]
        
        valid_corr_types = ["spearman", "pearson", "kendall", "linreg"]
        if type not in valid_corr_types:
            raise ValueError('unsupported type of correlation, must be one of: ' + ', '.join(valid_corr_types))
        else:
            self.type = type
    
    def __str__(self) -> str:
        return "PatternCorrelation with "+self.type
    
    def fit(self):
        if self.type == "spearman":
            self.result   = scipy.stats.spearmanr(self.X,self.Y).correlation
        elif self.type == "pearson":
            self.result   = scipy.stats.pearsonr(self.X,self.Y).correlation
        elif self.type == "kendall":
            self.result   = scipy.stats.kendalltau(self.X,self.Y).statistic
        elif self.type == "linreg":
            self.result   = scipy.stats.linregress(self.X,self.Y).slope
        return self
    
    def visualize(self):
        try:
            self.result
        except:
            self.fit()
        plot_models = [self.X,self.Y]
        plot_titles = ["neural rdm", "model rdm"]

        fig,axes = plt.subplots(1,2,figsize = (10,5))
        for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
            v = numpy.full(self.rdm_shape,numpy.nan)
            _,idx = lower_tri(v)
            v[idx] = m
            sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
            axes.flatten()[j].set_title(t)
        fig.suptitle(f'{self.type} correlation: {self.result}')
        return fig

class MultipleRDMRegression:
    def __init__(self,neuralrdm,modelrdms,modelnames:list=None) -> None:

        if modelnames is None:
            modelnames = ['m'+str(j) for j in range(len(modelrdms))]
        assert len(modelnames) == len(modelrdms), 'number of model names must be equal to number of model rdms'
        self.rdm_shape = neuralrdm.shape
        self.modelnames = modelnames

        Y,_ = lower_tri(neuralrdm)

        self.n_reg = len(modelrdms) # number of model rdms
        X = numpy.empty((len(Y),self.n_reg))
        for j,m in enumerate(modelrdms):
            X[:,j],_ = lower_tri(m)

        xna_filters = numpy.all([~numpy.isnan(X[:,j]) for j in range(numpy.shape(X)[1])],0)
        na_filters = numpy.logical_and(~numpy.isnan(Y),xna_filters)

        #standardize design matrix independently within each column
        X_dropNA = X[na_filters,:]
        self.X = standardize(X_dropNA,1)
        # standardize Y
        self.Y = standardize(Y[na_filters])
    
    def __str__(self) -> str:
        return "MultipleRDMRegression"
    
    def fit(self):
        reg = LinearRegression().fit(self.X,self.Y)
        self.reg = reg
        self.result = reg.coef_
        self.score  = reg.score(self.X,self.Y)
        #print(f'sklearn score: {reg.score(self.X,self.Y)}')
        #print(f'compute_R2: {compute_R2(reg.predict(self.X),self.Y,2)}')
        return self
    
    def visualize(self):
        try:
            self.result
        except:
            self.fit()
        plot_models = list(numpy.concatenate((numpy.atleast_2d(self.Y),self.X.T),axis=0))
        plot_titles = ["neural rdm"] + self.modelnames

        fig,axes = plt.subplots(1,len(plot_models),figsize = (10,5))
        for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
            v = numpy.full(self.rdm_shape,numpy.nan)
            _,idx = lower_tri(v)
            v[idx] = m
            sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
            if j == 0:
                axes.flatten()[j].set_title(t)
            else:
                axes.flatten()[j].set_title(f"{t}:{self.result[j-1]}")
        fig.suptitle(f'R2: {self.reg.score(self.X,self.Y)}')
        return fig