""" Class for estimators for RSA analysis
    All classes takes activity pattern matrix as an input and performs different types of RSA analysis based on the activity pattern matrix.
    An estimator class has at least four methods:
    (1) fit: by calling estimator.fit(), RSA analysis is performed, `result` attribute will be set. `estimator.result` is an 1D numpy array.
    (2) visualize: by calling estimator.visualize(), the result of RSA analysis will visualized, a figure handle will be returned
    (3) __str__: return the name of estimator class
    (4) get_details: return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON 
"""

import numpy
import scipy
from sklearn.linear_model import LinearRegression
from multivariate.helper import lower_tri, scale_feature, compute_rdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import time


def _get_pair_type_loc(dirpair_info_arr:numpy.array,dirpair_info_arr_cols:list)->dict:
    """classify coding direction pair type based on groundtruth location

    Parameters
    ----------
    dirpair_info_arr : numpy.array
        attribute of coding direction pair stored in a n_pairs * n_attributes numpy array
    dirpair_info_arr_cols : list
        names of the attributes (columns) in dirpair_info_arr

    Returns
    -------
    pair_type_dict:dict
        a dictionary. key is the name of the type of coding direction pairs, value is a (n_pairs,) boolean array used to filter the coding direction pairs that are classified as this type
    """
    sx_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="sx_dir1")[0]]
    sx_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="sx_dir2")[0]]
    sy_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="sy_dir1")[0]]
    sy_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="sy_dir2")[0]]
    ex_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ex_dir1")[0]]
    ex_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ex_dir2")[0]]
    ey_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ey_dir1")[0]]
    ey_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ey_dir2")[0]]

    same_sx = sx_dir1 == sx_dir2
    same_sy = sy_dir1 == sy_dir2
    same_ex = ex_dir1 == ex_dir2
    same_ey = ey_dir1 == ey_dir2

    sg_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="group1_dir1")[0]]
    eg_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="group2_dir1")[0]]
    sg_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="group1_dir2")[0]]
    eg_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="group2_dir2")[0]]
    traintrainpair = numpy.atleast_2d(numpy.all(numpy.hstack([sg_dir1==eg_dir1,sg_dir2==eg_dir2,sg_dir1==1,sg_dir2==1]),axis=1)).T
    testtestpair =  numpy.atleast_2d(numpy.all(numpy.hstack([sg_dir1==eg_dir1,sg_dir2==eg_dir2,sg_dir1==0,sg_dir2==0]),axis=1)).T
    mixpair =  numpy.atleast_2d(numpy.all(numpy.hstack([~traintrainpair,~testtestpair]),axis=1)).T
    
    pair_type_dict = {
        ## two coding directions that have the same start and end x but in different y rows
        "betweenX": numpy.all(
                        numpy.hstack([same_sx,same_ex,(sy_dir1 == ey_dir1),(sy_dir2 == ey_dir2)]),# a n_pairs*n_criteria array
                        axis=1),# compressed into (n_pairs,) array specifying which pair can be classified as betweenX, all criterial must be fullfiled for a pair to be classified as betweenX             
        ## two coding directions that have the same start and end y but in different x columns
        "betweenY": numpy.all(
                        numpy.hstack([same_sy,same_ey,(sx_dir1 == ex_dir1),(sx_dir2 == ex_dir2)]),
                        axis=1),
        ## two coding directions that are in the same x column
        "withinX": numpy.all(
                        numpy.hstack([ey_dir1 == sy_dir1,sy_dir2 == sy_dir1,ey_dir2 == sy_dir1]),
                        axis=1),                
        ## two coding directions that are in the same y row
        "withinY": numpy.all(
                        numpy.hstack([ex_dir1 == sx_dir1,sx_dir2 == sx_dir1,ex_dir2 == sx_dir1]),
                        axis=1),
        ################################ incorporate grouping into pair type classification ################################
        "withintrainX": numpy.all(
                        numpy.hstack([ey_dir1 == sy_dir1,sy_dir2 == sy_dir1,ey_dir2 == sy_dir1,traintrainpair]),
                        axis=1),
        "withintrainY": numpy.all(
                        numpy.hstack([ex_dir1 == sx_dir1,sx_dir2 == sx_dir1,ex_dir2 == sx_dir1,traintrainpair]),
                        axis=1),
        "withintestX": numpy.all(
                        numpy.hstack([ey_dir1 == sy_dir1,sy_dir2 == sy_dir1,ey_dir2 == sy_dir1,testtestpair]),
                        axis=1),
        "withintestY": numpy.all(
                        numpy.hstack([ex_dir1 == sx_dir1,sx_dir2 == sx_dir1,ex_dir2 == sx_dir1,testtestpair]),
                        axis=1), 
        "betweentraintestX": numpy.all(
                        numpy.hstack([same_sx,same_ex,(sy_dir1 == ey_dir1),(sy_dir2 == ey_dir2),mixpair]),
                        axis=1),
        "betweentraintestY": numpy.all(
                        numpy.hstack([same_sy,same_ey,(sx_dir1 == ex_dir1),(sx_dir2 == ex_dir2),mixpair]),
                        axis=1),
        "betweentestX": numpy.all(
                        numpy.hstack([same_sx,same_ex,(sy_dir1 == ey_dir1),(sy_dir2 == ey_dir2),testtestpair]),
                        axis=1),
        "betweentestY": numpy.all(
                        numpy.hstack([same_sy,same_ey,(sx_dir1 == ex_dir1),(sx_dir2 == ex_dir2),testtestpair]),
                        axis=1)              
    }
    return pair_type_dict
        
def _get_pair_type_feature(dirpair_info_arr:numpy.array,dirpair_info_arr_cols:list)->dict:
    """classify coding direction pair type based on visual features

    Parameters
    ----------
    dirpair_info_arr : numpy.array
        attribute of coding direction pair stored in a n_pairs * n_attributes numpy array
    dirpair_info_arr_cols : list
        names of the attributes (columns) in dirpair_info_arr

    Returns
    -------
    pair_type_dict:dict
        a dictionary. For each key-value pair, key is the name of the type of coding direction pairs, value is a (n_pairs,) boolean array used to filter the coding direction pairs that are classified as this type
    """
    sc_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="sc_dir1")[0]]
    sc_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="sc_dir2")[0]]
    ss_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ss_dir1")[0]]
    ss_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ss_dir2")[0]]
    ec_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ec_dir1")[0]]
    ec_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="ec_dir2")[0]]
    es_dir1 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="es_dir1")[0]]
    es_dir2 = dirpair_info_arr[:,numpy.where(numpy.array(dirpair_info_arr_cols)=="es_dir2")[0]]
    
    same_sc = sc_dir1 == sc_dir2
    same_ss = ss_dir1 == ss_dir2
    same_ec = ec_dir1 == ec_dir2
    same_es = es_dir1 == es_dir2

    pair_type_dict = {
        ## two coding directions that have the same start and end colours but in different shape rows
        "betweenColour": numpy.all(
                        numpy.hstack([same_sc,same_ec,ss_dir1 == es_dir1,ss_dir2 == es_dir2]),# a n_pairs*n_criteria array
                        axis=1),# compressed into (n_pairs,) array specifying which pair can be classified as betweenColour, all criterial must be fullfiled for a pair to be classified as betweenColour                
        ## two coding directions that have the same start and end shape but in different colour columns
        "betweenShape": numpy.all(
                        numpy.hstack([same_ss, same_es, sc_dir1 == ec_dir1, sc_dir2 == ec_dir2]),
                        axis=1),
        ## two coding directions that are in the same colour column
        "withinColour": numpy.all(
                        numpy.hstack([es_dir1 == ss_dir1, ss_dir2 == ss_dir1, es_dir2 == ss_dir1]),
                        axis=1),                
        ## two coding directions that are in the same shape row
        "withinShape": numpy.all(
                        numpy.hstack([ec_dir1 == sc_dir1, sc_dir2 == sc_dir1, ec_dir2 == sc_dir1]),
                        axis=1) 
    }
    return pair_type_dict


class NeuralDirectionCosineSimilarity:
    """class for running neural vector analysis. It calculates the cosine similarities in coding direction pairs.

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a n_sample*n_voxels activity pattern matrix
    stim_dict : dict
        dictionary of the information of each of the sample in the activity pattern matrix. Must include the following keys: stimid, stimsession, stimloc, stimfeature
    seed : int, optional
        the integer used as a random seed for random generator, by default None
    """
    def __init__(self,activitypattern:numpy.ndarray,stim_dict:dict,seed:int=None):
        self.X = activitypattern
        self.stimid = stim_dict["stimid"]
        self.stimsession = stim_dict["stimsession"]
        self.stimloc = stim_dict["stimloc"]
        self.stimfeature = stim_dict["stimfeature"]
        self.stimgroup = stim_dict["stimgroup"]
        self.seed = seed

    def fit(self):
        """running neural vector analysis. 
        
        The script first retrieve all possible pairs of coding directions and then classify the coding direction pairs into different pair types.
        For each pair, cosine similarity between the coding directions are computed. Then, the script aggregate within different pair type and return the mean values of all pair types.

        Meanwhile a permutated version of the activitiy matrix is generated, where voxels in one row is shuffled, and each row is shuffled independently. the above analysis is also performed with this shuffled data to use as control.

        """
        X = self.X
        #generate a randomly permutated X (randomly permutated within each row):
        randX = numpy.empty_like(X)
        for j in range(X.shape[0]):
            if self.seed is None:
                randX[j,:] = numpy.random.default_rng(seed=None).permutation(X.shape[1])
            else:
                randX[j,:] = numpy.random.default_rng(seed=self.seed+j*1000).permutation(X.shape[1])
       
        # compute coding directions between any two stimuli
        vidxs, uidxs = numpy.meshgrid(range(len(X)),range(len(X)))
        dirs_mat = X[uidxs]-X[vidxs]
        dirs_mat_rand = randX[uidxs]-randX[vidxs]
        dir_info_mat = numpy.dstack((
            self.stimid[uidxs],          self.stimid[vidxs],
            self.stimsession[uidxs],     self.stimsession[vidxs],
            self.stimgroup[uidxs],        self.stimgroup[vidxs],
            self.stimloc[:,0][uidxs],     self.stimloc[:,0][vidxs],
            self.stimloc[:,1][uidxs],     self.stimloc[:,1][vidxs],
            self.stimfeature[:,0][uidxs], self.stimfeature[:,0][vidxs],
            self.stimfeature[:,1][uidxs], self.stimfeature[:,1][vidxs]            
            ))
        dir_info_arr_cols = ['stim1', 'stim2','run1','run2','group1','group2', #stim id, run id and group id of stimulus 1 (starting stim) and 2 (ending stim)
                'sx','ex', #x location on groundtruth map for starting stim (sx) and ending stim (ex)
                'sy','ey', #y location on groundtruth map for starting stim (sy) and ending stim (ey)
                'sc','ec', #colour for starting stim (sc) and ending stim (ec)
                'ss','es'  #shape for starting stim (ss) and ending stim (es)
                ]

        dir_idx = numpy.tril_indices(dirs_mat.shape[0], k = -1)
        dirs_arr = dirs_mat[dir_idx]
        dirs_arr_rand = dirs_mat_rand[dir_idx]
        dir_info_arr = dir_info_mat[dir_idx] #dir_info_mat.reshape((-1,len(cols)))
        
        # get all possible pairs of coding directions
        dir1idxs, dir2idxs = numpy.meshgrid(range(dirs_arr.shape[0]),range(dirs_arr.shape[0]))
        dirpair_info_mat =  numpy.dstack((
            dir_info_arr[dir1idxs],dir_info_arr[dir2idxs]
            ))
        dirpair_idx = numpy.tril_indices(dirpair_info_mat.shape[0], k = -1)  
        dirpair_info_arr = dirpair_info_mat[dirpair_idx]
        dirpair_info_arr_cols = [x+'_dir1' for x in dir_info_arr_cols] + [x+'_dir2' for x in dir_info_arr_cols]
        dir1idxs_arr, dir2idxs_arr = dir1idxs[dirpair_idx], dir2idxs[dirpair_idx]
        
        # select a subset of pairs to do the computation
        pair_type_dict = {}
        pair_type_dict.update(
            _get_pair_type_loc(dirpair_info_arr,dirpair_info_arr_cols)
        )
        pair_type_dict.update(
            _get_pair_type_feature(dirpair_info_arr,dirpair_info_arr_cols)
        ) 
        
        # cosine similarity between pairs of coding directions
        cosine_similarity = lambda dir1, dir2: numpy.dot(dir1, dir2)/(numpy.linalg.norm(dir1)*numpy.linalg.norm(dir2))
        cos_similarity_neural = dict()
        cos_similarity_rand = dict()
        pairtypes=[]
        for k,v in pair_type_dict.items():
            if v.sum() > 0:
                cos_similarity_neural |= {
                    k:numpy.array([
                        cosine_similarity(dir1, dir2) for dir1,dir2 in zip(dirs_arr[dir1idxs_arr[v]],dirs_arr[dir2idxs_arr[v]])
                        ]).mean()
                }
                cos_similarity_rand |= {
                    k:numpy.array([
                        cosine_similarity(dir1, dir2) for dir1,dir2 in zip(dirs_arr_rand[dir1idxs_arr[v]],dirs_arr_rand[dir2idxs_arr[v]])
                        ]).mean()
                }
                pairtypes.append(k)
        
        if numpy.any(["between" in x for x in pairtypes]):
            cos_similarity_neural |= {
                "between": numpy.array([cos_similarity_neural["betweenX"],cos_similarity_neural["betweenY"]]).mean(),
                "betweentraintest": numpy.array([cos_similarity_neural["betweentraintestX"],cos_similarity_neural["betweentraintestY"]]).mean(),
                "betweentest": numpy.array([cos_similarity_neural["betweentestX"],cos_similarity_neural["betweentestY"]]).mean(),
                "within": numpy.array([cos_similarity_neural["withinX"],cos_similarity_neural["withinY"]]).mean(),
                "withintrain":numpy.array([cos_similarity_neural["withintrainX"],cos_similarity_neural["withintrainY"]]).mean(),
                "withintest":numpy.array([cos_similarity_neural["withintestX"],cos_similarity_neural["withintestY"]]).mean(),
                }
            cos_similarity_rand |= {
                "between": numpy.array([cos_similarity_rand["betweenX"],cos_similarity_rand["betweenY"]]).mean(),
                "betweentraintest": numpy.array([cos_similarity_rand["betweentraintestX"],cos_similarity_rand["betweentraintestY"]]).mean(),
                "betweentest": numpy.array([cos_similarity_rand["betweentestX"],cos_similarity_rand["betweentestY"]]).mean(),
                "within": numpy.array([cos_similarity_rand["withinX"],cos_similarity_rand["withinY"]]).mean(),
                "withintrain":numpy.array([cos_similarity_rand["withintrainX"],cos_similarity_rand["withintrainY"]]).mean(),
                "withintest":numpy.array([cos_similarity_rand["withintestX"],cos_similarity_rand["withintestY"]]).mean(),
                }
        else:
            cos_similarity_neural |= {
                "within": numpy.array([cos_similarity_neural["withinX"],cos_similarity_neural["withinY"]]).mean(),
                }
            cos_similarity_rand |= {
                "within": numpy.array([cos_similarity_rand["withinX"],cos_similarity_rand["withinY"]]).mean(),
                }
        pairtypes = list(cos_similarity_neural.keys())            
        
        self.resultdf = pandas.DataFrame({"pairtype":pairtypes,
                                          "neural":list(cos_similarity_neural.values()),
                                          "neuralrand":list(cos_similarity_rand.values())})

        #self.dir_pair_df = dir_pair_df
        self.result = numpy.array(list(cos_similarity_neural.values())+list(cos_similarity_rand.values()))
        self.resultnames = ['neuraldata_'+x for x in pairtypes] + ['shuffleddata_'+x for x in pairtypes]
        return self
    
    def visualize(self):
        try:
            self.resultdf
        except Exception:
            self.fit()        

        plotdf = pandas.melt(self.resultdf, id_vars=['pairtype'], value_vars=['neural','neuralrand'],
                         var_name='datatype', value_name='Cosine Similarity')
        plotdf["datatype"] = plotdf["datatype"].map({'neural': 'neural data', 'neuralrand': 'shuffled data'})

        fig = sns.catplot(data=plotdf,x="pairtype",y="Cosine Similarity",hue="datatype",kind="point")
        return fig
    

    def __str__(self) -> str:
        return f"NeuralDirectionCosineSimilarity"
    
    def get_details(self):
        details = {"name":self.__str__(),
                   "resultnames":self.resultnames,
                   "stim_dict":{"stimid":self.stimid.tolist(),
                                "stimsession":self.stimsession.tolist(),
                                "stimloc":self.stimloc.tolist(),
                                "stimfeature":self.stimfeature.tolist()}
                  }
        return  details


class PatternCorrelation:
    """calculate the correlation between neural rdm and model rdm

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a 2D numpy array. the neural activity pattern matrix used for computing representation dissimilarity matrix. size = (nsample,nfeatures)
    modelrdm : numpy.ndarray or list of numpy.ndarray
        a 2D numpy array. the dissimilarity matrices from different models of representation. size = (nsample,nsample)
    modelnames: list
        a list of model names. If `None` models will be named as m1,m2,..., by default `None`
    type : str, optional
        type of correlation measure, by default `"spearman"`.
        must be one of: `"spearman", "pearson", "kendall", "linreg"`
    """
    def __init__(self,activitypattern:numpy.ndarray,modelrdms:numpy.ndarray or list,modelnames:list=None,type:str="spearman",ztransform:bool=False) -> None:        

        #neural rdm
        neuralrdm = compute_rdm(activitypattern,"correlation")
        self.rdm_shape = neuralrdm.shape        
        self.Y,_ = lower_tri(neuralrdm)

        #model rdm and names        
        if isinstance(modelrdms,list):
            modelrdms = modelrdms
        elif isinstance(modelrdms,numpy.ndarrray):
            modelrdms = [modelrdms]
        else:
            raise TypeError('model rdm must be numpy ndarray or a list of numpy ndarray')
        
        if modelnames is None:
            modelnames = [f'm{str(j)}' for j in range(len(modelrdms))]            
        assert len(modelnames) == len(modelrdms), 'number of model names must be equal to number of model rdms'   

        self.Xs = []
        for m in modelrdms:
            X,_ = lower_tri(m)
            self.Xs.append(X)
        self.modelnames = modelnames  
        self.nX = len(modelrdms)

        #correlation type        
        valid_corr_types = ["spearman", "pearson", "kendall", "linreg"]
        if type not in valid_corr_types:
            raise ValueError('unsupported type of correlation, must be one of: ' + ', '.join(valid_corr_types))
        else:
            self.type = type
        self.outputtransform = lambda x: numpy.arctanh(x) if ztransform else x
    
    def fit(self):
        self.na_filters = []
        result = []
        for X in self.Xs:
            na_filters = numpy.logical_and(~numpy.isnan(self.Y),~numpy.isnan(X))
            self.na_filters.append(na_filters)
            Y = self.Y[na_filters]
            X = X[na_filters]
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

        fig,axes = plt.subplots(self.nX,2,figsize = (10,5*self.nX))
        for k,X in enumerate(self.Xs):
            plot_models = [self.Y,X]
            plot_titles = ["neural rdm", f"{self.modelnames[k]}"]
            for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
                v = numpy.full(self.rdm_shape,numpy.nan)
                _,idx = lower_tri(v)
                fillidx = (idx[0][self.na_filters[k]],idx[1][self.na_filters[k]])
                v[fillidx] = m[self.na_filters[k]]
                if self.nX==1:
                    sns.heatmap(v,ax=axes[j],square=True,cbar_kws={"shrink":0.85})
                    axes[j].set_title(t)
                else:
                    sns.heatmap(v,ax=axes[k][j],square=True,cbar_kws={"shrink":0.85})
                    axes[k][j].set_title(t)
        fig.suptitle(f'{self.type} correlation: {self.result}')
        return fig
    
    def __str__(self) -> str:
        return f"PatternCorrelation with {self.type}"
    
    def get_details(self):        
        details = {"name":self.__str__(),
                   "corrtype":self.type,
                   "NAfilters":dict(zip(self.modelnames,[x.tolist() for x in self.na_filters])),
                   "modelRDMs":dict(zip(self.modelnames,[x.tolist() for x in self.Xs]))
                  }
        return  details

class MultipleRDMRegression:
    """estimate the regression coefficient when using the model rdms to predict neural rdm

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a 2D numpy array. the neural activity pattern matrix used for computing representation dissimilarity matrix. size = (nsample,nfeatures)
    modelrdm : numpy.ndarray or list of numpy.ndarray
        a 2D numpy array. the dissimilarity matrices from different models of representation. size = (nsample,nsample)
    modelnames: list
        a list of model names. If `None` models will be named as m1,m2,..., by default `None`
    """
    def __init__(self,activitypattern:numpy.ndarray,modelrdms:numpy.ndarray or list,modelnames:list=None,standardize:bool=True) -> None:

        #model rdm and names        
        if isinstance(modelrdms,list):
            modelrdms = modelrdms
        elif isinstance(modelrdms,numpy.ndarrray):
            modelrdms = [modelrdms]
        else:
            raise TypeError('model rdm must be numpy ndarray or a list of numpy ndarray')

        if modelnames is None:
            modelnames = [f'm{str(j)}' for j in range(len(modelrdms))]
        assert len(modelnames) == len(modelrdms), 'number of model names must be equal to number of model rdms'

        #neural rdm
        neuralrdm = compute_rdm(activitypattern,"correlation")
        self.rdm_shape = neuralrdm.shape
        self.modelnames = modelnames

        Y,_ = lower_tri(neuralrdm)

        self.n_reg = len(modelrdms) # number of model rdms
        X = numpy.empty((len(Y),self.n_reg)) # X is a nvoxel * nmodel matrix
        for j,m in enumerate(modelrdms):
            X[:,j],_ = lower_tri(m)

        self.X = X
        self.Y = Y

        self.standardize = standardize
    
    def fit(self):
        xna_filters = numpy.all(~numpy.isnan(self.X),1) # find out rows that are not nans in all columns
        self.na_filters = numpy.logical_and(~numpy.isnan(self.Y),xna_filters)

        if self.standardize:
            #standardize design matrix independently within each column
            X = scale_feature(self.X[self.na_filters,:],1)
            # standardize Y
            Y = scale_feature(self.Y[self.na_filters])
        else:
            X = self.X[self.na_filters,:]
            Y = self.Y[self.na_filters]

        reg = LinearRegression().fit(X,Y)
        self.reg = reg
        self.result = reg.coef_
        self.score  = reg.score(self.X,self.Y)
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
            v[fillidx] = m[self.na_filters]
            sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
            if j == 0:
                axes.flatten()[j].set_title(t)
            else:
                axes.flatten()[j].set_title(f"{t}:{self.result[j-1]}")
        fig.suptitle(f'R2: {self.score}')
        return fig

    def __str__(self) -> str:
        return "MultipleRDMRegression"
    
    def get_details(self):        
        details = {"name":self.__str__(),
                   "standardize":self.standardize*1,
                   "NAfilters":self.na_filters.tolist(),
                   "modelRDMs":dict(zip(self.modelnames,[x.tolist() for x in self.X.T]))
                  }
        return  details
    
