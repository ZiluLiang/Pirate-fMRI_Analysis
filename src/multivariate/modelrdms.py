"""
This module contain the class to generate model RDM of the pirate project

Zilu Liang @HIPlab Oxford
2023
"""
import scipy
import numpy
import pandas
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from typing import Union

from zpyhelper.MVPA.rdm import compute_rdm, compute_rdm_nomial, compute_rdm_identity, lower_tri, upper_tri

class ModelRDM:
    """ set up the model rdms based on stimuli properties in pirate exp

    Parameters
    ----------
    stimid : numpy.ndarray
        a 1D/2D numpy array of stimuli ids, shape = `(nstim,)` or `(nstim,1)`
    stimgtloc : numpy.ndarray
        a 2D numpy array of stimuli groundtruth locations (x and y), shape = `(nstim,2)`
    stimfeature : numpy.ndarray
        a 2D numpy array of stimuli features (color and feature), shape = `(nstim,2)`
    stimgroup : numpy.ndarray
        a 1D/2D numpy array of stimuli group, shape = `(nstim,)` or `(nstim,1)`
    sessions : int, optional
        a 1D/2D numpy array of stimuli sessions, shape = `(nstim,)` or `(nstim,1)`
        If multiple sessions are present, model rdms will be separated in to between and within case as well. by default 1
    stimresploc : numpy.ndarray
        a 2D numpy array of stimuli locations (x and y) based on participants response, shape = `(nstim*n_session,2)`, by default None
        number of sessions, by default 1
    randomseed: int, optional
        randomseed passed to `self.random` to initiate random generator to create random model rdm, by default 1
    nan_identity: bool, optional
        whether or not to nan out same-same stimuli pairs in the model rdm. by default true.  
    splitgroup: bool, optional
        whether or not to split into train-train and test-test pairs. by default false.    
    """
    
    def __init__(self,
                 stimid:numpy.ndarray,
                 stimgtloc:numpy.ndarray,
                 stimfeature:numpy.ndarray,
                 stimgroup:numpy.ndarray,
                 sessions:numpy.array,
                 stimresploc:numpy.array=None,
                 nan_identity:bool=True,
                 splitgroup:bool=False):
        
        self.n_session   = numpy.unique(sessions).size
        self.stimid      = numpy.atleast_2d(stimid).T if numpy.ndim(stimid) <2 else numpy.array(stimid)
        self.n_stim      = self.stimid.shape[0]

        #### stimuli features
        ## groundtruth location - 2D        
        self.stimloc     = numpy.array(stimgtloc)
        
        ## 'hierarchical' model - 4D: quadrant (global feature) - location within each quadrant (local feature)
        # local feature encoded using central axis as reference
        self.stimloc_wrtcentre = numpy.concatenate(
                                    [numpy.absolute(self.stimloc),
                                     numpy.sign(self.stimloc)
                                    ],axis=1) 
        # local features are encoded as from left-right and bottom-up, centre locations are encoded as zero
        wrtlrbu_x,wrtlrbu_y = self.stimloc[:,[0]],self.stimloc[:,[1]]
        wrtlrbu_x[wrtlrbu_x==0]=numpy.nan
        wrtlrbu_y[wrtlrbu_y==0]=numpy.nan
        wrtlrbu_x = scipy.stats.rankdata(wrtlrbu_x,method="dense",axis=0,nan_policy="omit")
        wrtlrbu_y = scipy.stats.rankdata(wrtlrbu_y,method="dense",axis=0,nan_policy="omit")
        wrtlrbu_x[numpy.isnan(wrtlrbu_x)]=0
        wrtlrbu_y[numpy.isnan(wrtlrbu_y)]=0
        self.stimloc_wrtlrbu  = numpy.concatenate([wrtlrbu_x, wrtlrbu_y, self.stimloc_wrtcentre[:,[2,3]]],axis=1)
        
        ## two-hot encoded color/shape features - 10D
        self.stimfeature = numpy.array(stimfeature)

        # stimuli group and session
        self.stimgroup   = numpy.atleast_2d(stimgroup).T if numpy.ndim(stimgroup) <2 else numpy.array(stimgroup) 
        self.stimsession = numpy.atleast_2d(sessions).T if numpy.ndim(sessions) <2 else numpy.array(sessions) 
        
        ## response location
        if stimresploc is None:
            self.gen_resprdm = False
            self.stimresploc,self.resploc_wrtcentre = numpy.full_like(self.stimloc,fill_value=numpy.nan),numpy.full_like(self.stimloc_wrtcentre,fill_value=numpy.nan)
            
        else:
            self.gen_resprdm = True
            self.stimresploc     = stimresploc
            self.resploc_wrtcentre = numpy.concatenate(
                                    [numpy.absolute(self.stimresploc),
                                     numpy.sign(self.stimresploc)
                                    ],axis=1)
        
        #concatenate into a df for convenience ## TODO: maybe use a structured numpy array? which is more efficient?
        self.stimdf = pandas.DataFrame(
            {
                "stim_id":self.stimid[:,0],
                "stim_group":self.stimgroup[:,0],
                "stim_session":self.stimsession[:,0],
                "stim_x":self.stimloc[:,0],
                "stim_y":self.stimloc[:,1],
                "stim_color":self.stimfeature[:,0],
                "stim_shape":self.stimfeature[:,1],                
                "stim_xsign":self.stimloc_wrtcentre[:,2],
                "stim_ysign":self.stimloc_wrtcentre[:,3],
                "stim_xdist":self.stimloc_wrtcentre[:,0],            
                "stim_ydist":self.stimloc_wrtcentre[:,1],
                "stim_lrbux":self.stimloc_wrtlrbu[:,0],
                "stim_lrbuy":self.stimloc_wrtlrbu[:,1],
                "resp_x":self.stimresploc[:,0],
                "resp_y":self.stimresploc[:,1],
                "resp_xsign":self.resploc_wrtcentre[:,2],
                "resp_ysign":self.resploc_wrtcentre[:,3],
                "resp_xdist":self.resploc_wrtcentre[:,0],            
                "resp_ydist":self.resploc_wrtcentre[:,1]
            },                
        )
        
        self.splitgroup = splitgroup
        self.nan_identity = nan_identity
        
        self.models = self.genModels()

    def genModels(self,shuffle_idx=None,shuffle_columns=[])->dict:
        """generate model rdms

        Parameters
        ----------
        shuffle_idx : array, optional
            indices for shuffling rows of the stimuli df before generating model rdm, if `None`, no shuffling will be applied
            by default None
        shuffle_columns : list, optional
            columns to apply the shuffling, if empty (`[]`), no shuffling will be applied
            allowed values: `['stim_id', 'stim_group', 'stim_session', 'stim_x', 'stim_y',
                              'stim_color', 'stim_shape', 'stim_xsign', 'stim_ysign', 'stim_xdist',
                              'stim_ydist', 'stim_lrbux', 'stim_lrbuy', 'resp_x', 'resp_y',
                              'resp_xsign', 'resp_ysign', 'resp_xdist', 'resp_ydist',
                              'gtloc','feature2d','stim_xysign','stim_xydist','stim_lrbuxy',
                              'resploc','resploc_xydist','resploc_xysign']`
            by default []

        Returns
        -------
        dictionary
            a dictionary of model rdms
        """

        if shuffle_idx is None:
            shuffle_idx = numpy.arange(self.n_stim*self.n_session)
        else:
            assert (numpy.unique(shuffle_idx)==numpy.arange(self.n_stim*self.n_session)).all()
        
        allow_shuffle_cols = ['stim_id', 'stim_group', 'stim_session', 'stim_x', 'stim_y',
                              'stim_color', 'stim_shape', 'stim_xsign', 'stim_ysign', 'stim_xdist',
                              'stim_ydist', 'stim_lrbux', 'stim_lrbuy', 'resp_x', 'resp_y',
                              'resp_xsign', 'resp_ysign', 'resp_xdist', 'resp_ydist',
                              'gtloc','feature2d','stim_xysign','stim_xydist','stim_lrbuxy',
                              'resploc','resploc_xydist','resploc_xysign']
        assert all([x in allow_shuffle_cols for x in shuffle_columns]), f"{numpy.array(shuffle_columns)[[x not in allow_shuffle_cols for x in shuffle_columns]]} are not valid column names for shuffling"
        # retrieve feature arrays for model rdm construction
        stim = {}
        
        ### 1d features
        stim = dict(zip(self.stimdf.columns, self.stimdf.to_numpy().T))
        for k,v in stim.items():
            stim[k] = numpy.atleast_2d(v).T

        ### 2d features
        stim["gtloc"]          = numpy.array(self.stimdf[["stim_x","stim_y"]])
        stim["feature2d"]      = numpy.array(self.stimdf[["stim_color","stim_shape"]])
        stim["stim_xysign"]    = numpy.array(self.stimdf[["stim_xsign","stim_ysign"]])
        stim["stim_xydist"]    = numpy.array(self.stimdf[["stim_xdist","stim_ydist"]])
        stim["stim_lrbuxy"]    = numpy.array(self.stimdf[["stim_lrbux","stim_lrbuy"]])
        stim["resploc"]        = numpy.array(self.stimdf[["resp_x","resp_y"]])
        stim["resploc_xydist"] = numpy.array(self.stimdf[["resp_xdist","resp_ydist"]])
        stim["resploc_xysign"] = numpy.array(self.stimdf[["resp_xsign","resp_ysign"]])
        
        
        ### shuffle
        for k,v in stim.items():
            if k in shuffle_columns:
                stim[k] = v[shuffle_idx,:]
        
        #### model RDMs
        models = {}

        euclidean_rdms = dict(zip(
                              [# flat models
                               "gtlocEuclidean","gtloc1dx","gtloc1dy",
                               # 'hierachical' models: global + local feature
                               "global_xsign","global_ysign","global_xysign", ## global features                               
                               "locwrtcentre_localx","locwrtcentre_localy","locwrtcentre_localxy",## local feature encoded using central axis as reference - gt
                               "locwrtlrbu_localx","locwrtlrbu_localy","locwrtlrbu_localxy"  ## local features are encoded as from left-right and bottom-up
                               ],
                              ["gtloc","stim_x","stim_y",
                               "stim_xsign","stim_ysign","stim_xysign",
                               "stim_xdist","stim_ydist","stim_xydist",
                               "stim_lrbux","stim_lrbuy","stim_lrbuxy"])) 
        for mname,col in euclidean_rdms.items():
            models[mname] = compute_rdm(stim[col],metric="euclidean")

        nomial_rdms = dict(zip(["feature2d"],["feature2d"]))
        for mname,col in nomial_rdms.items():
            models[mname] = compute_rdm_nomial(stim[col].squeeze())

        identity_rdms = dict(zip(
                            ["feature1d_color","feature1d_shape","stimuli","stimuligroup"],
                            ["stim_color","stim_shape","stim_id","stim_group"]
        ))
        for mname,col in identity_rdms.items():
            models[mname] = compute_rdm_identity(stim[col].squeeze())

        if self.gen_resprdm:
            euclidean_rdms = dict(zip(
                              [# flat models
                               "resplocEuclidean","resploc1dx","resploc1dy",
                               # 'hierachical' models: global + local feature
                               "respglobal_xsign","respglobal_ysign","respglobal_xysign", ## global features                               
                               "resplocwrtcentre_localx","resplocwrtcentre_localy","resplocwrtcentre_localxy"
                               ],
                              ["resploc","resp_x","resp_y",
                               "resp_xsign","resp_ysign","resploc_xysign",
                               "resp_xdist","resp_ydist","resploc_xydist"]
            )) 
            for mname,col in euclidean_rdms.items():
                models[mname] = compute_rdm(stim[col],metric="euclidean")


        # split rdm into train/test/mix pairs
        if numpy.logical_and(self.splitgroup,numpy.unique(self.stimgroup).size>1):
            U,V = numpy.meshgrid(self.stimgroup,self.stimgroup)
            WTR = numpy.multiply(1.*(U == 1),1.*(V == 1))
            WTE = numpy.multiply(1.*(U == 0),1.*(V == 0))
            MIX = 1. * ~(U==V)
            WTR[WTR==0]=numpy.nan
            WTE[WTE==0]=numpy.nan
            MIX[MIX==0]=numpy.nan
            split_models = [x for x in models.keys() if x not in ["stimuli","stimuligroup"]]
            for k in split_models:
                wtr_n = 'trainstimpairs_' + k
                rdmwtr = numpy.multiply(models[k],WTR)
                wte_n = 'teststimpairs_' + k
                rdmwte = numpy.multiply(models[k],WTE)
                mix_n = 'mixedstimpairs_' + k
                rdmmix = numpy.multiply(models[k],MIX)
                models |= {wtr_n:rdmwtr,wte_n:rdmwte,mix_n:rdmmix}

        # split into sessions
        if self.n_session>1:
            BS = compute_rdm_identity(self.stimsession) # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=numpy.nan
            WS[WS==0]=numpy.nan
            tmp = list(models.items()) ## this is so that we don't change models
            for k,v in tmp:
                ws_n  = 'within_'+k
                rdmws = numpy.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = numpy.multiply(v,BS)
                models |= {ws_n:rdmws,bs_n:rdmbs}
            models["session"] = compute_rdm_identity(self.stimsession)
        for k,v in models.items():
            if self.nan_identity:
                from copy import deepcopy
                nan_matrix = deepcopy(models["stimuli"])
                nan_matrix[numpy.where(nan_matrix==0)] = numpy.nan
                models[k] = numpy.multiply(v,nan_matrix)
        
        # remove models in which lower tri have only one value that is not nan
        valid_mn = [k for k,v in models.items() if numpy.sum(~numpy.isnan(numpy.unique(lower_tri(v)[0])))>1]
        valid_m = [models[k] for k in valid_mn]
        models = dict(zip(valid_mn,valid_m))
        return models


    def __str__(self):
        return 'The following model rdms are created:\n' + ',\n'.join(
            self.models.keys()
        )
      

    def visualize(self,modelname:Union[str,list]="all",tri:int=0,annot:bool=False)->matplotlib.figure:
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
    
    def rdm_to_df(self,modelnames:Union[list,str],rdms:Union[list,numpy.ndarray]=None) -> pandas.DataFrame:
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
            df = pandas.DataFrame({'stimidA':numpy.squeeze(self.stimid[c]),    'stimidB': numpy.squeeze(self.stimid[i]), 
                                   'groupA': numpy.squeeze(self.stimgroup[c]), 'groupB': numpy.squeeze(self.stimgroup[i]), 
                                   'runA': numpy.squeeze(self.stimsession[c]), 'runB': numpy.squeeze(self.stimsession[i]), 
                                    m: lt}
                                 ).set_index(['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB'])
            modeldf.append(df)
        modeldf = modeldf[0].join(modeldf[1:])
        return modeldf

"""double check by running the following code:
import itertools
f = [0.,1.,2.,3.,4.]
stimfeature = numpy.array(list(itertools.product(f,f)))
stimgtloc = stimfeature-2
stimid = numpy.arange(stimgtloc.shape[0])
stimgroup = numpy.array([all([x==0,y==0])*1 for [x,y] in stimgtloc])
nsess = 2
stimfeature = numpy.tile(stimfeature,(nsess,1))
stimgtloc = numpy.tile(stimgtloc,(nsess,1))
stimid = numpy.tile(numpy.atleast_2d(stimid).T,(nsess,1))
stimgroup = numpy.tile(numpy.atleast_2d(stimgroup).T,(nsess,1))
sessions = numpy.concatenate([numpy.ones((25,1))*x for x in range(nsess)],axis=0)
modelrdm = ModelRDM(stimid,stimgtloc,stimfeature,stimgroup,sessions,splitgroup=True)
violate_keys = ["within_session","within_stimuli","teststimpairs_stimuligroup"]
non_const_check = all([x not in modelrdm.models.keys() for x in violate_keys])

non_const_check must return True
"""

