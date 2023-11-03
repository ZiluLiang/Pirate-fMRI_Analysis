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

from helper import compute_rdm, compute_rdm_nomial, compute_rdm_identity, lower_tri, upper_tri

class ModelRDM:
    """ set up the model rdms based on stimuli properties in pirate exp

    Parameters
    ----------
    stimid : numpy.ndarray
        a 1D numpy array of stimuli ids
    stimgtloc : numpy.ndarray
        a 2D numpy array of stimuli groundtruth locations (x and y), shape = `(nstim,2)`
    stimfeature : numpy.ndarray
        a 2D numpy array of stimuli features (color and feature), shape = `(nstim,2)`
    stimwrttrainloc : numpy.ndarray
        a 2D numpy array of stimuli locations with regard to training location, shape = `(nstim,4)`, the four columns corresponds to: xdist, ydist, xsign, ysign
    stimresploc : numpy.ndarray
        a 2D numpy array of stimuli locations (x and y) based on participants response, shape = `(nstim,2)`, by default None
        number of sessions, by default 1
    n_session : int, optional
        number of sessions, stimuli matrix will be repeated by number of sessions before constructing rdm.
        If multiple sessions are present, model rdms will be separated in to between and within case as well. by default 1
    randomseed: int, optional
        randomseed passed to `self.random` to initiate random generator to create random model rdm, by default 1
    nan_identity: bool, optional
        whether or not to nan out same-same stimuli pairs in the model rdm. by default true.    
    """
    
    def __init__(self,
                 stimid:numpy.ndarray,
                 stimgtloc:numpy.ndarray,
                 stimfeature:numpy.ndarray,
                 stimgroup:numpy.ndarray,
                 stimresploc:numpy.array=None,
                 n_session:int=1,
                 randomseed:int=1,
                 nan_identity:bool=True,
                 splitgroup:bool=False):
        
        self.n_session   = n_session
        self.n_stim      = len(stimid)       
        self.stimid      = numpy.tile(stimid,(n_session,1)).reshape((self.n_stim*self.n_session,-1))

        #### stimuli features
        ## groundtruth location - 2D        
        self.stimloc     = numpy.tile(stimgtloc,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        
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
        
        ## one-hot encoded color/shape features - 10D
        self.stimfeature = numpy.tile(stimfeature,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimgroup   = numpy.tile(stimgroup,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        self.stimsession = numpy.tile(stimid,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
        
        ## response location
        if stimresploc is None:
            gen_resprdm = False
            self.stimresploc,self.resploc_wrtcentre = numpy.full_like(self.stimloc,fill_value=numpy.nan),numpy.full_like(self.stimloc_wrtcentre,fill_value=numpy.nan)
            
        else:
            gen_resprdm = True
            
            # if response of all four sessions are availble, or if using the concatenated one run and the averaged response map
            if numpy.shape(stimresploc)[0] == self.stimloc.shape[0]:
                self.stimresploc     = stimresploc
            # else stack the averaged response map
            else:
                self.stimresploc     = numpy.tile(stimresploc,(n_session,1)).reshape((self.n_stim*self.n_session,-1))
            self.resploc_wrtcentre = numpy.concatenate(
                                    [numpy.absolute(self.stimresploc),
                                     numpy.sign(self.stimresploc)
                                    ],axis=1)
        
        #concatenate into a df for convenience ## TODO: maybe use a structured numpy array? which is more efficient?
        self.stimdf = pandas.DataFrame(
            {
                "stim_id":self.stimid,
                "stim_group":self.stimgroup,
                "stim_session":self.stimsession,
                "stim_x":self.stimloc[:,0],
                "stim_y":self.stimloc[:,1],
                "stim_xsign":self.stimloc_wrtcentre[:,2],
                "stim_ysign":self.stimloc_wrtcentre[:,3],
                "stim_xdist":self.stimloc_wrtcentre[:,0],            
                "stim_ydist":self.stimloc_wrtcentre[:,1],
                "resp_x":self.stimresploc[:,0],
                "resp_y":self.stimresploc[:,1],
                "resp_xsign":self.resploc_wrtcentre[:,2],
                "resp_ysign":self.resploc_wrtcentre[:,3],
                "resp_xdist":self.resploc_wrtcentre[:,0],            
                "resp_ydist":self.resploc_wrtcentre[:,1]
            }                
        )
        
        #### model RDMs
        models = {"gtlocEuclidean":   compute_rdm(self.stimloc,metric="euclidean"),
                  "gtlocCityBlock":   compute_rdm(self.stimloc,metric="cityblock"),
                  "gtloc1dx":         compute_rdm(self.stimloc[:,[0]],metric="euclidean"),
                  "gtloc1dy":         compute_rdm(self.stimloc[:,[1]],metric="euclidean"),
                  "feature2d":        compute_rdm_nomial(self.stimfeature),
                  "feature1dx":       compute_rdm_identity(self.stimfeature[:,0]),
                  "feature1dy":       compute_rdm_identity(self.stimfeature[:,1]),
                  "stimuli":          compute_rdm_identity(self.stimid),
                  "stimuligroup":     compute_rdm_identity(self.stimgroup),
                  ## 'hierachical' models: global + local feature
                  "global_xsign":      compute_rdm(self.stimloc_wrtcentre[:,[2]],metric="euclidean"),
                  "global_ysign":      compute_rdm(self.stimloc_wrtcentre[:,[3]],metric="euclidean"),
                  "global_xysign":     compute_rdm(self.stimloc_wrtcentre[:,[2,3]],metric="euclidean"),
                  # local feature encoded using central axis as reference - gt
                  "locwrtcentre_localglobal": compute_rdm(self.stimloc_wrtcentre,metric="euclidean"),
                  "locwrtcentre_localx":      compute_rdm(self.stimloc_wrtcentre[:,[0]],metric="euclidean"),
                  "locwrtcentre_localy":      compute_rdm(self.stimloc_wrtcentre[:,[1]],metric="euclidean"),
                  "locwrtcentre_localxy":     compute_rdm(self.stimloc_wrtcentre[:,[0,1]],metric="euclidean"),
                  # local features are encoded as from left-right and bottom-up
                  "locwrtlrbu_localglobal":   compute_rdm(self.stimloc_wrtlrbu,metric="euclidean"),
                  "locwrtlrbu_localx":      compute_rdm(self.stimloc_wrtlrbu[:,[0]],metric="euclidean"),
                  "locwrtlrbu_localy":      compute_rdm(self.stimloc_wrtlrbu[:,[1]],metric="euclidean"),
                  "locwrtlrbu_localxy":     compute_rdm(self.stimloc_wrtlrbu[:,[0,1]],metric="euclidean"),

                  ## 'hierachical' models: global + local feature - resp
                  "respglobal_xsign":      compute_rdm(self.resploc_wrtcentre[:,[2]],metric="euclidean"),
                  "respglobal_ysign":      compute_rdm(self.resploc_wrtcentre[:,[3]],metric="euclidean"),
                  "respglobal_xysign":     compute_rdm(self.resploc_wrtcentre[:,[2,3]],metric="euclidean"),
                  # local feature encoded using central axis as reference - resp
                  "resplocwrtcentre_localglobal": compute_rdm(self.resploc_wrtcentre,metric="euclidean"),
                  "resplocwrtcentre_localx":      compute_rdm(self.resploc_wrtcentre[:,[0]],metric="euclidean"),
                  "resplocwrtcentre_localy":      compute_rdm(self.resploc_wrtcentre[:,[1]],metric="euclidean"),
                  "resplocwrtcentre_localxy":     compute_rdm(self.resploc_wrtcentre[:,[0,1]],metric="euclidean"),
                  }
        models |= {
                  "shuffledloc2d": self.random(randomseed=randomseed,rdm=models["gtlocEuclidean"],mode="permuterdm"),
#                  "randfeature2d": self.random(randomseed=randomseed,mode="randomfeature"),
#                  "randmatrix":    self.random(randomseed=randomseed,mode="randommatrix")
                }
        
        if gen_resprdm:
            models |= {"resplocEuclidean":  compute_rdm(self.stimresploc,metric="euclidean"),
                       "resplocCityBlock":  compute_rdm(self.stimresploc,metric="cityblock"),
                       "resploc1dx":        compute_rdm(self.stimresploc[:,[0]],metric="euclidean"),
                       "resploc1dy":        compute_rdm(self.stimresploc[:,[1]],metric="euclidean")}

        # split rdm into train/test/mix pairs
        if numpy.logical_and(splitgroup,numpy.unique(self.stimgroup).size>1):
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
        if n_session>1:
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
            if nan_identity:
                from copy import deepcopy
                nan_matrix = deepcopy(models["stimuli"])
                nan_matrix[numpy.where(nan_matrix==0)] = numpy.nan
                models[k] = numpy.multiply(v,nan_matrix)
        self.models = models


    def __str__(self):
        return 'The following model rdms are created:\n' + ',\n'.join(
            self.models.keys()
        )
    
    def random(self,randomseed:int=1,rdm:numpy.ndarray=None,mode:str="randommatrix") -> numpy.ndarray:
        """generate random model RDM based on sample size

        Parameters
        ----------
        randomseed: int
            a random seed used for the random state instance that are used for generating random matrix/feature/permutation
        rdm : numpy.ndarray
            a 2D numpy array of a model rdm that is used for shuffling to generate random RDM
        mode : str
            a string specifying the kind of random rdm to be generated. Must be one of: randomfeature, permuterdm,randommatrix

        Returns
        -------
        numpy.ndarray
            2D numpy array of a random model rdm
        """
        
        prng = numpy.random.RandomState(randomseed)
        if rdm is not None:
            assert isinstance(rdm,numpy.ndarray), "rdm must be 2D square array of size = (nsample,nsample)"
            assert rdm.shape == (self.n_stim*self.n_session,self.n_stim*self.n_session), "rdm must be 2D square array of size = (nsample,nsample)"
            mode = "permuterdm"

        if mode == "permuterdm":
            return prng.permutation(rdm.flatten()).reshape(rdm.shape)
        elif mode == "randomfeature":
            rand_features = prng.random(size=self.stimfeature.shape)
            return compute_rdm(rand_features,"euclidean")
        elif mode == "randommatrix":
            return prng.random((self.n_stim*self.n_session,self.n_stim*self.n_session))
        else:
            raise ValueError("invalid random mode, must be one of: randomfeature, permuterdm, randommatrix")    

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
    
    def rdm_to_df(self,modelnames:list or str,rdms:list or numpy.ndarray=None) -> pandas.DataFrame:
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
            df = pandas.DataFrame({'stimidA':numpy.squeeze(self.stimid[c]), 'stimidB': numpy.squeeze(self.stimid[i]), 
                                   'groupA': numpy.squeeze(self.stimgroup[c]),  'groupB': numpy.squeeze(self.stimgroup[i]), 
                                   'runA': numpy.squeeze(self.stimsession[c]),  'runB': numpy.squeeze(self.stimsession[i]), 
                                    m: lt}
                                 ).set_index(['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB'])
            modeldf.append(df)
        modeldf = modeldf[0].join(modeldf[1:])
        return modeldf
        