""" Estimator classes for RSA analysis
    All classes takes activity pattern matrix as an input and performs different types of RSA analysis based on the activity pattern matrix.
    An estimator class has at least four methods:
    (1) fit: by calling estimator.fit(), RSA analysis is performed, `result` attribute will be set. `estimator.result` is an 1D numpy array.
    (2) visualize: by calling estimator.visualize(), the result of RSA analysis will visualized, a figure handle will be returned
    (3) __str__: return the name of estimator class
    (4) get_details: return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON 

Zilu Liang @HIPlab Oxford
2023
"""

import numpy
import scipy
from sklearn.linear_model import LinearRegression
from multivariate.helper import lower_tri, scale_feature, compute_rdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import itertools
import time
from sklearn.svm import SVR,SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler



def _get_pair_type_loc(dirpair_info:numpy.array,dirpair_info_cols:list)->dict:
    """classify coding direction pair type based on groundtruth location

    Parameters
    ----------
    dirpair_info : numpy.array
        attribute of coding direction pair stored in a n_pairs * n_attributes numpy array
    dirpair_info_cols : dict
        column indexnames of the attributes (columns) in dirpair_info_arr

    Returns
    -------
    pair_type_dict:dict
        a dictionary. key is the name of the type of coding direction pairs, value is a (n_pairs,) boolean array used to filter the coding direction pairs that are classified as this type
    """
    #filter out directions that are samestimuli-samestimuli directions
    
    #starting x/y of each direction
    sx_dir1 = dirpair_info[:,dirpair_info_cols["sx_dir1"]]
    sx_dir2 = dirpair_info[:,dirpair_info_cols["sx_dir2"]]
    sy_dir1 = dirpair_info[:,dirpair_info_cols["sy_dir1"]]
    sy_dir2 = dirpair_info[:,dirpair_info_cols["sy_dir2"]]        
    #ending x/y of each direction 
    ex_dir1 = dirpair_info[:,dirpair_info_cols["ex_dir1"]]
    ex_dir2 = dirpair_info[:,dirpair_info_cols["ex_dir2"]]
    ey_dir1 = dirpair_info[:,dirpair_info_cols["ey_dir1"]]
    ey_dir2 = dirpair_info[:,dirpair_info_cols["ey_dir2"]]
    #whether the direction pair shares starting/ending x/y
    same_sx = sx_dir1 == sx_dir2
    same_sy = sy_dir1 == sy_dir2
    same_ex = ex_dir1 == ex_dir2
    same_ey = ey_dir1 == ey_dir2

    #sign of starting x/y of each direction 
    sxs_dir1 = dirpair_info[:,dirpair_info_cols["sxs_dir1"]]
    sxs_dir2 = dirpair_info[:,dirpair_info_cols["sxs_dir2"]]
    sys_dir1 = dirpair_info[:,dirpair_info_cols["sys_dir1"]]
    sys_dir2 = dirpair_info[:,dirpair_info_cols["sys_dir2"]]        
    #sign of ending x/y of each direction 
    exs_dir1 = dirpair_info[:,dirpair_info_cols["exs_dir1"]]
    exs_dir2 = dirpair_info[:,dirpair_info_cols["exs_dir2"]]
    eys_dir1 = dirpair_info[:,dirpair_info_cols["eys_dir1"]]
    eys_dir2 = dirpair_info[:,dirpair_info_cols["eys_dir2"]]
    #whether the starting and ending point of direction is in the same quadrant
    sesame_xs_dir1 = sxs_dir1 == exs_dir1 # true: same quafrant; false: different quadrant
    sesame_ys_dir1 = sys_dir1 == eys_dir1
    sesame_xs_dir2 = sxs_dir2 == exs_dir2
    sesame_ys_dir2 = sys_dir2 == eys_dir2
    
    # distance
    sxd_dir1 = dirpair_info[:,dirpair_info_cols["sxd_dir1"]]
    sxd_dir2 = dirpair_info[:,dirpair_info_cols["sxd_dir2"]]
    syd_dir1 = dirpair_info[:,dirpair_info_cols["syd_dir1"]]
    syd_dir2 = dirpair_info[:,dirpair_info_cols["syd_dir2"]]        
    exd_dir1 = dirpair_info[:,dirpair_info_cols["exd_dir1"]]
    exd_dir2 = dirpair_info[:,dirpair_info_cols["exd_dir2"]]
    eyd_dir1 = dirpair_info[:,dirpair_info_cols["eyd_dir1"]]
    eyd_dir2 = dirpair_info[:,dirpair_info_cols["eyd_dir2"]]
    sesame_xd_dir1 = sxd_dir1 == exd_dir1 # true: same distance to centre; false: different dist
    sesame_yd_dir1 = syd_dir1 == eyd_dir1
    sesame_xd_dir2 = sxd_dir2 == exd_dir2
    sesame_yd_dir2 = syd_dir2 == eyd_dir2
    # 
    cqcq_x = numpy.atleast_2d(numpy.all(numpy.hstack([sesame_xs_dir1==sesame_xs_dir2,~sesame_xs_dir1,sesame_xd_dir1,sesame_xd_dir2]),axis=1)).T # cross quadrant x
    cqcq_y = numpy.atleast_2d(numpy.all(numpy.hstack([sesame_ys_dir1==sesame_ys_dir2,~sesame_ys_dir1,sesame_yd_dir1,sesame_yd_dir2]),axis=1)).T # cross quadrant x
    cqwq_x = numpy.atleast_2d(numpy.all(numpy.hstack([sesame_xs_dir1!=sesame_xs_dir2,sesame_xd_dir1!=sesame_xd_dir2]),axis=1)).T
    cqwq_y = numpy.atleast_2d(numpy.all(numpy.hstack([sesame_ys_dir1!=sesame_ys_dir2,sesame_yd_dir1!=sesame_yd_dir2]),axis=1)).T

    # group
    sg_dir1 = dirpair_info[:,dirpair_info_cols["sg_dir1"]]
    eg_dir1 = dirpair_info[:,dirpair_info_cols["eg_dir1"]]
    sg_dir2 = dirpair_info[:,dirpair_info_cols["sg_dir2"]]
    eg_dir2 = dirpair_info[:,dirpair_info_cols["eg_dir2"]]
    traintrainpair = numpy.atleast_2d(numpy.all(numpy.hstack([sg_dir1==eg_dir1,sg_dir2==eg_dir2,sg_dir1==1,sg_dir2==1]),axis=1)).T
    testtestpair =  numpy.atleast_2d(numpy.all(numpy.hstack([sg_dir1==eg_dir1,sg_dir2==eg_dir2,sg_dir1==0,sg_dir2==0]),axis=1)).T
    mixgrouppair =  numpy.atleast_2d(numpy.all(numpy.hstack([~traintrainpair,~testtestpair]),axis=1)).T


    #session of starting and ending locations
    sr_dir1 = dirpair_info[:,dirpair_info_cols["sr_dir1"]]
    er_dir1 = dirpair_info[:,dirpair_info_cols["er_dir1"]]
    sr_dir2 = dirpair_info[:,dirpair_info_cols["sr_dir2"]]
    er_dir2 = dirpair_info[:,dirpair_info_cols["er_dir2"]]
    withinrunpair = numpy.atleast_2d(numpy.all(numpy.hstack([sr_dir1==er_dir1,sr_dir2==er_dir2,sr_dir1==sr_dir2]),axis=1)).T
    betweenrunpair =  numpy.atleast_2d(numpy.all(numpy.hstack([sr_dir1==er_dir1,sr_dir2==er_dir2,sr_dir1!=sr_dir2]),axis=1)).T
    
    pair_type_dict = {
        "wrun_withinX_cqcq": numpy.all(
                        numpy.hstack([ey_dir1 == sy_dir1,sy_dir2 == sy_dir1,ey_dir2 == sy_dir1,
                                      #~same_sx, ~same_ex, ex_dir1 != sx_dir2, ex_dir2 != sx_dir1, 
                                      withinrunpair, cqcq_x, testtestpair]),
                        axis=1),
        "wrun_withinX_cqwq": numpy.all(
                        numpy.hstack([ey_dir1 == sy_dir1,sy_dir2 == sy_dir1,ey_dir2 == sy_dir1,
                                      #~same_sx, ~same_ex, ex_dir1 != sx_dir2, ex_dir2 != sx_dir1, 
                                      withinrunpair, cqwq_x, testtestpair]),
                        axis=1),
        "wrun_withinY_cqcq": numpy.all(
                        numpy.hstack([ex_dir1 == sx_dir1,sx_dir2 == sx_dir1,ex_dir2 == sx_dir1,
                                      #~same_sy, ~same_ey, ey_dir1 != sy_dir2, ey_dir2 != sy_dir1,
                                      withinrunpair, cqcq_y, testtestpair]),
                        axis=1),
        "wrun_withinY_cqwq": numpy.all(
                        numpy.hstack([ex_dir1 == sx_dir1,sx_dir2 == sx_dir1,ex_dir2 == sx_dir1,
                                      #~same_sy, ~same_ey, ey_dir1 != sy_dir2, ey_dir2 != sy_dir1,
                                      withinrunpair, cqwq_y, testtestpair]),
                        axis=1),
        
                        
        "wrun_betweenX": numpy.all(
                        numpy.hstack([same_sx,same_ex,(sy_dir1 == ey_dir1),(sy_dir2 == ey_dir2), (sy_dir1 != sy_dir2), withinrunpair]),
                        axis=1),
        "wrun_betweenY": numpy.all(
                        numpy.hstack([same_sy,same_ey,(sx_dir1 == ex_dir1),(sx_dir2 == ex_dir2), (sx_dir1 != sx_dir2), withinrunpair]),
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
        dir_idx = numpy.tril_indices(uidxs.shape[0], k = -1)
        vidxs, uidxs = vidxs[dir_idx],uidxs[dir_idx]        
        dirs_arr = X[uidxs]-X[vidxs]
        dirs_arr_rand = randX[uidxs]-randX[vidxs]
        dir_info = numpy.dstack((
            self.stimid[uidxs],          self.stimid[vidxs],
            self.stimloc[:,0][uidxs],     self.stimloc[:,0][vidxs],
            self.stimloc[:,1][uidxs],     self.stimloc[:,1][vidxs],
            self.stimfeature[:,0][uidxs], self.stimfeature[:,0][vidxs],
            self.stimfeature[:,1][uidxs], self.stimfeature[:,1][vidxs],            
            self.stimsession[uidxs],     self.stimsession[vidxs],
            self.stimgroup[uidxs],        self.stimgroup[vidxs],            
            numpy.sign(self.stimloc[:,0][uidxs]),     numpy.sign(self.stimloc[:,0][vidxs]),
            numpy.sign(self.stimloc[:,1][uidxs]),     numpy.sign(self.stimloc[:,1][vidxs]),            
            numpy.absolute(self.stimloc[:,0][uidxs]),     numpy.absolute(self.stimloc[:,0][vidxs]),
            numpy.absolute(self.stimloc[:,1][uidxs]),     numpy.absolute(self.stimloc[:,1][vidxs])
            )).squeeze()
        dir_info_cols = ['sid', 'eid', #stim id of starting stim and ending stim
                         'sx','ex', # x location on groundtruth map for starting stim (sx) and ending stim (ex)
                         'sy','ey', # y location on groundtruth map for starting stim (sy) and ending stim (ey)
                         'sc','ec', # colour for starting stim (sc) and ending stim (ec)
                         'ss','es',  # shape for starting stim (ss) and ending stim (es)
                         'sr','er', # run(session) number for starting stim (sr) and ending stim (er)
                         'sg','eg',  # group(train vs test) for starting stim (sg) and ending stim (eg)
                         'sxs','exs', # x sign on groundtruth map for starting stim (sx) and ending stim (ex)
                         'sys','eys', # y sign on groundtruth map for starting stim (sx) and ending stim (ex)
                         'sxd','exd', # x sign on groundtruth map for starting stim (sx) and ending stim (ex)
                         'syd','eyd' # y sign on groundtruth map for starting stim (sx) and ending stim (ex)
                        ]

        
        # get all possible pairs of coding directions
        dir1idxs, dir2idxs = numpy.meshgrid(range(dirs_arr.shape[0]),range(dirs_arr.shape[0]))
        dirpair_idx = numpy.tril_indices(dir1idxs.shape[0], k = -1)
        dirpair_info =  numpy.hstack((
            dir_info[dir1idxs[dirpair_idx]],dir_info[dir2idxs[dirpair_idx]]
            ))
        colsname = [x+'_dir1' for x in dir_info_cols] + [x+'_dir2' for x in dir_info_cols]
        dirpair_info_cols = dict(zip(
            colsname,
            [numpy.where(numpy.array(colsname)==k)[0] for k in colsname]
        ))
        sid_dir1= dirpair_info[:,dirpair_info_cols["sid_dir1"]]
        eid_dir1= dirpair_info[:,dirpair_info_cols["eid_dir1"]]
        sid_dir2= dirpair_info[:,dirpair_info_cols["sid_dir2"]]
        eid_dir2= dirpair_info[:,dirpair_info_cols["eid_dir2"]]
        same_sx = dirpair_info[:,dirpair_info_cols["sx_dir1"]] == dirpair_info[:,dirpair_info_cols["sx_dir2"]]
        same_sy = dirpair_info[:,dirpair_info_cols["sy_dir1"]] == dirpair_info[:,dirpair_info_cols["sy_dir2"]]
        same_ex = dirpair_info[:,dirpair_info_cols["ex_dir1"]] == dirpair_info[:,dirpair_info_cols["ex_dir2"]]
        same_ey = dirpair_info[:,dirpair_info_cols["ey_dir1"]] == dirpair_info[:,dirpair_info_cols["ey_dir2"]]
        withinrun_dir1 = dirpair_info[:,dirpair_info_cols["sr_dir1"]] == dirpair_info[:,dirpair_info_cols["er_dir1"]]
        withinrun_dir2 = dirpair_info[:,dirpair_info_cols["sr_dir2"]] == dirpair_info[:,dirpair_info_cols["er_dir2"]]
        withinrunpairs = dirpair_info[:,dirpair_info_cols["sr_dir1"]] == dirpair_info[:,dirpair_info_cols["sr_dir2"]]
        includepairs = numpy.all(numpy.hstack([
            sid_dir1!=eid_dir1,
            sid_dir2!=eid_dir2,
            withinrun_dir1,
            withinrun_dir2,
            withinrunpairs,
            ~(same_sx&same_ex&same_sy&same_ey)]),axis=1)
        dir1idxs, dir2idxs = dir1idxs[dirpair_idx][includepairs], dir2idxs[dirpair_idx][includepairs]
        dirpair_info = dirpair_info[includepairs]
            
        # select a subset of pairs to do the computation
        pair_type_dict = {}
        pair_type_dict.update(
            _get_pair_type_loc(dirpair_info,dirpair_info_cols)
        )

        # cosine similarity between pairs of coding direction        
        cosine_similarity = lambda dir1, dir2: numpy.dot(dir1, dir2)/(numpy.linalg.norm(dir1)*numpy.linalg.norm(dir2))
        cos_similarity_neural = dict()
        cos_similarity_rand = dict()
        pairtypes=[]
        for k,v in pair_type_dict.items():
            if v.sum() > 0:
                cos_similarity_neural |= {
                    k:numpy.array([
                        cosine_similarity(dir1, dir2) for dir1,dir2 in zip(dirs_arr[dir1idxs[v]],dirs_arr[dir2idxs[v]])
                        ]).mean()
                }
                cos_similarity_rand |= {
                    k:numpy.array([
                        cosine_similarity(dir1, dir2) for dir1,dir2 in zip(dirs_arr_rand[dir1idxs[v]],dirs_arr_rand[dir2idxs[v]])
                        ]).mean()
                }
                pairtypes.append(k)                    
        
        pairtypes = list(cos_similarity_neural.keys())            
        
        self.resultdf = pandas.DataFrame({"pairtype":pairtypes,
                                          "neural":list(cos_similarity_neural.values()),
                                          "neuralrand":list(cos_similarity_rand.values())})

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
    rdm_metric: str, optional
        dissimilarity/distance metric passed to `scipy.spatial.distance.pdist` to compute neural rdm from activity pattern matrix, by default `"correlation"`
    type : str, optional
        type of correlation measure, by default `"spearman"`.
        must be one of: `"spearman", "pearson", "kendall", "linreg"`
    ztransform: bool, optional
        whether or not to perform fisher Z transform to the correlation coefficients, by default `none`.    
    """
    def __init__(self,activitypattern:numpy.ndarray,modelrdms:numpy.ndarray or list,modelnames:list=None,rdm_metric:str="correlation",
                 type:str="spearman",ztransform:bool=False) -> None:        

        #neural rdm
        neuralrdm = compute_rdm(activitypattern,rdm_metric)
        self.rdm_shape = neuralrdm.shape        
        self.Y,_ = lower_tri(neuralrdm)

        #model rdm and names        
        if isinstance(modelrdms,list):
            modelrdms = modelrdms
        elif isinstance(modelrdms,numpy.ndarray):
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
    modelrdms : numpy.ndarray or list of numpy.ndarray
        a 2D numpy array. the dissimilarity matrices from different models of representation. size = (nsample,nsample)
    modelnames: list, optional
        a list of model names. If `None` models will be named as m1,m2,..., by default `None`
    standardize: bool, optional
        whether or not to standardize the model rdms and neural rdms before running regression, by default `None`
    """
    def __init__(self,activitypattern:numpy.ndarray,modelrdms:numpy.ndarray or list,modelnames:list=None,standardize:bool=True) -> None:

        #model rdm and names        
        if isinstance(modelrdms,list):
            modelrdms = modelrdms
        elif isinstance(modelrdms,numpy.ndarray):
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
        self.score  = reg.score(X,Y)
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
                   "modelRDMs":dict(zip(self.modelnames,[x.tolist() for x in self.X.T])),
                   "score":self.score
                  }
        return  details
    
class SVMDecoder:
    """class for running decoding analysis. It decodes the x/y attribute in a discrete (will be extended to continuous case in the future) manner using Support Vector Machine.

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a n_sample*n_voxels activity pattern matrix
    stim_dict : dict
        dictionary of the information of each of the sample in the activity pattern matrix. Must include the following keys: stimid, stimsession, stimloc, stimfeature
    categorical_decoder: bool, optional
        whether decoder should do continuous or categorical output
    PCA_component: float, optional
        Controller for performing PCA before doing decoding analysis. It is the n_component argument passed to `PCA` class from sklearn.decomposition module. It specifies the number of PCA components(when n_component>1) or ratio of variance explained(when n_component<1) when performing PCA on the activity pattern matrix.
    seed : int, optional
        the integer used as a random seed for random generator, by default None
    """
    def __init__(self,activitypattern:numpy.ndarray,stim_dict:dict,categorical_decoder:bool=True,PCA_component:float=None,seed:int=None):
        
        self.stimid = stim_dict["stimid"]
        self.stimsession = stim_dict["stimsession"]
        self.stimloc = stim_dict["stimloc"]
        self.stimfeature = stim_dict["stimfeature"]
        self.stimgroup = stim_dict["stimgroup"]
        self.seed = seed
        self.categorical_decoder = categorical_decoder
        self.PCA_component = PCA_component
        
        if self.PCA_component is None:
            self.X = activitypattern
        else:            
            self.X = PCA(n_components=self.PCA_component).fit_transform(activitypattern)

    def fit(self):
        """running decoding analysis. 
        """
        APM = self.X
        runs = self.stimsession
        labels = {
            "x":self.stimfeature[:,0],
            "y":self.stimfeature[:,1]
        }
        evaluation_scores = {"x":[],"y":[]}
        training_scores = {"x":[],"y":[]}
        #rc_list = [list(x) for x in itertools.combinations([0,1,2,3,4], 2)] + [[x] for x in [0,1,2,3,4]]
        rc_list = [[[2],[2]]]
        for rc in rc_list:
            for curr_ax in ["x","y"]:
                if curr_ax == "x":
                    traintestsplit = numpy.in1d(self.stimfeature[:,1], rc)*1
                else:
                    traintestsplit = numpy.in1d(self.stimfeature[:,0], rc)*1
                if self.categorical_decoder:
                    clf = SVC(max_iter=10000, tol=1e-3, 
                            C = 0.1,
                            kernel='linear',
                            random_state = self.seed)
                else:
                    clf = SVR(max_iter=10000, tol=1e-3, 
                            C = 0.1,
                            kernel='linear')
                tmp_scores = cross_validate(estimator = clf,
                                        X = StandardScaler().fit_transform(APM),
                                        y = labels[curr_ax],
                                        groups = traintestsplit,
                                        cv = LeaveOneGroupOut(),
                                        return_train_score=True)
                training_scores[curr_ax].append(tmp_scores['train_score'])
                evaluation_scores[curr_ax].append(tmp_scores['test_score'])
        subset_scores = numpy.concatenate(list(training_scores.values()) + list(evaluation_scores.values()),axis=1).squeeze()
        average_scores = [numpy.array(v).mean() for v in training_scores.values()] + [numpy.array(v).mean() for v in evaluation_scores.values()]
        subset_scorenames = sum(
            [[f"training_trainstim{x}",f"training_teststim{x}"] for x in training_scores.keys()] + [[f"evaluation_teststim{x}",f"evaluation_trainstim{x}"] for x in evaluation_scores.keys()],
            [])
        average_scorenames = [f"training_{x}" for x in training_scores.keys()] + [f"evaluation_{x}" for x in evaluation_scores.keys()]
        self.result = numpy.concatenate((subset_scores,average_scores))
        self.resultnames = sum([subset_scorenames,average_scorenames],[])
        return self
    
    def visualize(self):
        pass
    

    def __str__(self) -> str:
        return "SVMDecoder"
    
    def get_details(self):
        details = {"name":self.__str__(),
                   "resultnames":self.resultnames,
                   "PCA_component":self.PCA_component,
                   "stim_dict":{"stimid":self.stimid.tolist(),
                                "stimsession":self.stimsession.tolist(),
                                "stimloc":self.stimloc.tolist(),
                                "stimfeature":self.stimfeature.tolist()}
                  }
        return  details