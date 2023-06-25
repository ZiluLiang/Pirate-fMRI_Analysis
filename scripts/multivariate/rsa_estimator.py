import numpy
import scipy
from sklearn.linear_model import LinearRegression
from multivariate.helper import lower_tri, upper_tri, scale_feature, compute_R2
import matplotlib.pyplot as plt
import seaborn as sns

class PatternCorrelation:
    """calculate the correlation between neural rdm and model rdm

    Parameters
    ----------
    neuralrdm : numpy.ndarray
        a 2D numpy array. the neural representation dissimilarity matrix. 
    modelrdm : numpy.ndarray or list of numpy.ndarray
        a 2D numpy array. the model representation dissimilarity matrix. 
    type : str, optional
        type of correlation measure, by default "spearman".
        must be one of: "spearman", "pearson", "kendall", "linreg"
    """
    def __init__(self,neuralrdm:numpy.ndarray,modelrdm:numpy.ndarray or list,type:str="spearman",ztransform:bool=False) -> None:
        self.rdm_shape = neuralrdm.shape
        
        self.X,_ = lower_tri(neuralrdm)
        if isinstance(modelrdm,list):
            modelrdm = modelrdm
        elif isinstance(modelrdm,numpy.ndarrray):
            modelrdm = [modelrdm]
            Y,_ = lower_tri(modelrdm)
        else:
            raise TypeError('model rdm must be numpy ndarray or a list of numpy ndarray')
        
        self.Ys = []
        for m in modelrdm:
            Y,_ = lower_tri(m)
            self.Ys.append(Y)
        self.nY = len(modelrdm)
        
        valid_corr_types = ["spearman", "pearson", "kendall", "linreg"]
        if type not in valid_corr_types:
            raise ValueError('unsupported type of correlation, must be one of: ' + ', '.join(valid_corr_types))
        else:
            self.type = type
        self.outputtransform = lambda x: numpy.arctanh(x) if ztransform else x

    
    def __str__(self) -> str:
        return f"PatternCorrelation with {self.type}"
    
    def fit(self):
        self.na_filters = []
        result = []
        for Y in self.Ys:
            na_filters = numpy.logical_and(~numpy.isnan(self.X),~numpy.isnan(Y))
            self.na_filters.append(na_filters)
            X = self.X[na_filters]
            Y = Y[na_filters]

            if self.type == "spearman":
                r = self.outputtransform(scipy.stats.spearmanr(X,Y).correlation)
            elif self.type == "pearson":
                r = self.outputtransform(scipy.stats.pearsonr(X,Y).correlation)
            elif self.type == "kendall":
                r = scipy.stats.kendalltau(X,Y).statistic
            elif self.type == "linreg":
                r = scipy.stats.linregress(X,Y).slope
            result.append(r)
        self.result = numpy.array(result)
        return self
    
    def visualize(self):
        try:
            self.result
        except Exception:
            self.fit()        

        fig,axes = plt.subplots(self.nY,2,figsize = (10,5*self.nY))
        for k,Y in enumerate(self.Ys):
            plot_models = [self.X,Y]
            plot_titles = ["neural rdm", f"model rdm {k}"]
            for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
                v = numpy.full(self.rdm_shape,numpy.nan)
                _,idx = lower_tri(v)
                fillidx = (idx[0][self.na_filters[k]],idx[1][self.na_filters[k]])
                v[fillidx] = m[self.na_filters[k]]
                sns.heatmap(v,ax=axes[k][j],square=True,cbar_kws={"shrink":0.85})
                axes[k][j].set_title(t)
        fig.suptitle(f'{self.type} correlation: {self.result}')
        return fig

class MultipleRDMRegression:
    def __init__(self,neuralrdm,modelrdms,modelnames:list=None) -> None:

        if modelnames is None:
            modelnames = [f'm{str(j)}' for j in range(len(modelrdms))]
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
        self.na_filters = na_filters

        #standardize design matrix independently within each column
        X_dropNA = X[na_filters,:]
        self.X = scale_feature(X_dropNA,1)
        # standardize Y
        self.Y = scale_feature(Y[na_filters])
    
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
        except Exception:
            self.fit()
        plot_models = list(numpy.concatenate((numpy.atleast_2d(self.Y),self.X.T),axis=0))
        plot_titles = ["neural rdm"] + self.modelnames

        fig,axes = plt.subplots(1,len(plot_models),figsize = (10,5))
        for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
            v = numpy.full(self.rdm_shape,numpy.nan)
            _,idx = lower_tri(v)
            fillidx = (idx[0][self.na_filters],idx[1][self.na_filters])
            v[fillidx] = m
            sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
            if j == 0:
                axes.flatten()[j].set_title(t)
            else:
                axes.flatten()[j].set_title(f"{t}:{self.result[j-1]}")
        fig.suptitle(f'R2: {self.reg.score(self.X,self.Y)}')
        return fig