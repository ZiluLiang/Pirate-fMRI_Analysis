"""
This module contains class that wraps up RSA analysis in ROI or whole brain into a pipeline.
The runner class is dependent on the file structure as specified in `FILESTRUCTURE.md`

Zilu Liang @HIPlab Oxford
2023
"""

import itertools
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import glob
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump
from sklearn.manifold import MDS   
from sklearn.decomposition import PCA

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import compute_rdm, checkdir, scale_feature
from multivariate.modelrdms import ModelRDM
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression,NeuralDirectionCosineSimilarity,SVMDecoder

class RSARunner:
    """
    The `RSARunner` wraps up RSA analysis in ROI and whole brain into a pipeline.

    Parameters
    ----------
    participants : list
        list of participant ids
    fmribeh_dir : str
        the parent directory that holds participants' folder of behavioral data
    nsession : int
        number of sessions/runs 
    beta_dir : list
        the parent directory that holds participants' folder of beta images that are used to construct the activity pattern matrix
    beta_fname : list
        _the name of the beta image in each participant's folder
    vsmask_dir : list
        the parent directory that holds participants' folder of voxel selection masks
    vsmask_fname : list
        the name of the voxel selection masks in each participant's folder
    pmask_dir : list, optional
        the parent directory that holds participants' folder of process masks , by default None
    pmask_fname : list, optional
        the name of process masks in each participant's folder, by default None
    anatmasks : list, optional
        the path to anatomical masks (not participant specific), by default None
    taskname : str, optional
        name of the task, by default "localizer"
    config_modelrdm: dict, optional
        configurations for computing model rdm
    config_neuralrdm: dict, optional
        configurations for computing neural rdm
    """

    def __init__(self,participants:list,
                 fmribeh_dir:str,
                 nsession:int,
                 beta_dir:list,beta_fname:list,
                 vsmask_dir:list,vsmask_fname:list,
                 pmask_dir:list=None,pmask_fname:list=None,
                 anatmasks:list=None,
                 taskname:str="localizer",
                 config_modelrdm:dict={"randomseed":None,"nan_identity":True,"splitgroup":False},
                 config_neuralrdm:dict={"preproc":None,"distance_metric":"correlation"}) -> None:
        """
        set up the paths to the image and set up other configurations
        """	
        
        self.participants = participants # participant list
        self.fmribeh_dir  = fmribeh_dir  # behavioral data directory to look for stimuli information

        # diretory and file names of activity pattern image:
        self.beta_dir     = beta_dir
        self.beta_fname   = beta_fname
        self.res_fname    = []
        # diretory and file names of voxel selection masks: participant specific masks of voxels included in computing neural rdm
        self.vsmask_dir   = vsmask_dir
        self.vsmask_fname = vsmask_fname
        # diretory and file names of process mask:  participant-specific masks of voxels used to generate searchlight spheres
        self.pmask_dir    = vsmask_dir if pmask_dir is None else pmask_dir
        self.pmask_fname  = vsmask_fname if pmask_fname is None else pmask_fname
        self.anatmasks    = anatmasks  # anatomical masks: non-participant-specific masks

        # task
        assert taskname in {"localizer", "navigation"}, "invalid task name!"
        self.taskname = taskname
        
        # number of sessions
        self.nsession     = nsession

        # options
        self.config_modelrdm  = config_modelrdm
        self.config_neuralrdm = config_neuralrdm

############################################### PARTICIPANT-SPECIFIC DATA EXTRACTION METHODS ###################################################
    def get_imagedir(self,subid):
        beta_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.beta_dir,self.beta_fname)]
        mask_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.vsmask_dir,self.vsmask_fname)]
        pmask_imgs = [os.path.join(d,'first',subid,f) for d,f in zip(self.pmask_dir,self.pmask_fname)]
        res_imgs = [os.path.join(d,'first',subid,f) for d,f in zip(self.beta_dir,self.res_fname)]
        return beta_imgs,mask_imgs,pmask_imgs
    
    def get_modelRDM(self,subid):
        fmribeh_dir = self.fmribeh_dir
        # load stimuli list
        stim_list_fn = glob.glob(os.path.join(fmribeh_dir,subid,'sub*_stimlist.txt'))[0]
        stim_list    =  pd.read_csv(stim_list_fn, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False)# use `sort_values` to make sure stim list is in the same order for all participants
        
        stim_id = np.array(stim_list['stim_id']) # get stimuli id
        stim_image = np.array([x.replace('.png','') for x in stim_list["stim_img"]]) # get stimuli image
        stim_train = np.array(stim_list['training']) # get training/test stimuli classification
        
        # get 2d location
        stim_locori = np.array(stim_list[['stim_x','stim_y']])
        old_range = np.max(stim_locori) - np.min(stim_locori)
        new_range = 1-(-1)
        stim_loc = (new_range*(stim_locori-np.min(stim_locori))/old_range) - 1

        # get visual features
        stim_color = np.array([x.replace('.png','').split('_')[0] for x in stim_list["stim_img"]])
        stim_shape = np.array([x.replace('.png','').split('_')[1] for x in stim_list["stim_img"]])
        _, stim_color = np.unique(stim_color, return_inverse=True) # recode string into integer
        _, stim_shape = np.unique(stim_shape, return_inverse=True) # recode string into integer
        stim_feature = np.vstack([stim_color,stim_shape]).T

        filter = np.where(stim_train==1) if self.taskname == "localizer" else (np.arange(stim_train.size),)
        stimid,stimgtloc,stimfeature,stimgroup =  stim_id[filter],stim_loc[filter],stim_feature[filter],stim_train[filter]    

        # load behavioral response
        participant_resp_fns = glob.glob(os.path.join(fmribeh_dir,subid,'sub*_task-piratenavigation_run-*.csv'))
        resp_lists = [pd.read_csv(x, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False).iloc[filter[0],:] for x in participant_resp_fns]
        if self.nsession == 4:
            resp_df_runs = pd.concat(resp_lists,axis=0)
            stimresploc_ori = np.array(resp_df_runs[['resp_x','resp_y']])
        elif self.nsession ==1:
            stimresploc_ori = np.mean([np.array(x[['resp_x','resp_y']]) for x in resp_lists],axis=0)
        elif self.nsession ==2:
            stimresploc_odd  = np.mean([np.array(x[['resp_x','resp_y']]) for x in [resp_lists[0],resp_lists[2]]],axis=0)
            stimresploc_even = np.mean([np.array(x[['resp_x','resp_y']]) for x in [resp_lists[1],resp_lists[3]]],axis=0)
            stimresploc_ori = np.concatenate([stimresploc_odd,stimresploc_even],axis=0)
        # rescale response location the same way as stimuli location
        stimresploc = (new_range*(stimresploc_ori-np.min(stim_locori))/old_range) - 1

        return ModelRDM(stimid      = stimid,
                        stimgtloc   = stimgtloc,
                        stimfeature = stimfeature,
                        stimgroup   = stimgroup,
                        stimresploc = stimresploc,
                        n_session   = self.nsession,
                        **self.config_modelrdm)

    def get_neuralRDM(self,subid):
        ##TODO: set up different preprocessing of activity pattern matrix before generating neural RDM
        ## model rdm
        modelrdm = self.get_modelRDM(subid)

        ## get neural data
        beta_imgs,vs_masks,_ = self.get_imagedir(subid)
        mask_imgs = vs_masks + self.anatmasks
        APD = ActivityPatternDataLoader(beta_imgs,mask_imgs)
        
        ## preprocessing
        ev = 1
        preproc=self.config_neuralrdm["preproc"]
        if preproc is None:
            activitypattern = APD.X
        elif preproc == "cocktail_blank_removal":
            for k in modelrdm.stimgroup:
                activitypattern[np.where(modelrdm.stimgroup == k),:] = scale_feature(activitypattern[np.where(modelrdm.stimgroup == k),:],1,False)
        elif preproc =="subtract_mean":
            activitypattern = scale_feature(activitypattern,1,False)
        elif preproc =="PCA":
            neuralPCA = PCA(n_components=0.9)
            activitypattern = neuralPCA.fit_transform(APD.X)
            ev = np.sum(neuralPCA.fit(APD.X).explained_variance_ratio_)
            
        return activitypattern,compute_rdm(activitypattern,self.config_neuralrdm["distance_metric"]),ev
        
############################################### SINGLE PARTICIPANT LEVEL ANALSYSIS METHODS ###################################################
    def _singleparticipant_ROIRSA(self, subid,
               corr_rdm_names=None,
               corr_type="spearman",
               randomseed:int=1,nan_identity:bool=True,
               verbose:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## compute model rdm
        modelrdm  = self.get_modelRDM(subid)        

        ### choose which model rdm is used for calculating correlation
        if corr_rdm_names is None:
            corr_rdm_names = [x for x in modelrdm.models.keys() if not np.any([x.endswith('session'),x.endswith('stimuli')])]
        else:
            corr_rdm_names = [x for x in modelrdm.models.keys() if x in corr_rdm_names]
        corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]
        ### put model rdm into dataframe
        df_inc_models = [x for x in corr_rdm_names  if not np.logical_or(x.startswith('between_'),x.startswith('within_'))]
        modeldf = modelrdm.rdm_to_df(df_inc_models)

        ## get neural data
        activitypattern, neural_rdm, _ = self.get_neuralRDM(subid)
        neuralrdmdf = modelrdm.rdm_to_df(modelnames="neural",rdms=neural_rdm)
        neuralrdmdf = neuralrdmdf.join(modeldf).reset_index().assign(subid=subid)
        
        ## compute correlations between neural rdm and model rdms
        if isinstance(corr_type,str):
            corr_type = [corr_type]
        elif not isinstance(corr_type,list):
            raise ValueError('invalid input type')
        
        corr_df = []
        for cr in corr_type:
            PC_s =  PatternCorrelation(activitypattern = activitypattern,
                                    modelrdms = corr_rdm_vals,
                                    modelnames = corr_rdm_names,
                                    type=cr)
            PC_s.fit()
            corr_name_dict = dict(zip(range(len(corr_rdm_vals)),corr_rdm_names))
            cr_df = pd.DataFrame(PC_s.result).T.rename(columns = corr_name_dict).assign(analysis=cr,subid=subid)
            corr_df.append(cr_df)
        return pd.concat(corr_df,axis=0),neuralrdmdf
    
    def _singleparticipant_ROIMDS(self, subid,
                n_components:int=2,
                randomseed:int=1,
                verbose:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## get neural data
        activitypattern, _, _ = self.get_neuralRDM(subid)
        ## compute MDS
        embedding = MDS(
                n_components=n_components,
                max_iter=5000,
                n_init=100,
                n_jobs=1,
                normalized_stress=False,
                random_state=randomseed,
                )
        X_transformed = embedding.fit_transform(activitypattern)
        axis_names = [f'axis {j}' for j in range(X_transformed.shape[1])]
        X_df = pd.DataFrame(X_transformed,columns=axis_names)
        modelrdm  = self.get_modelRDM(subid)
        mds_df = pd.DataFrame({"stim_id":modelrdm.stimid.flatten(),
                                "stim_x":modelrdm.stimloc[:,0],
                                "stim_y":modelrdm.stimloc[:,1],
                                "stim_color":modelrdm.stimfeature[:,0],
                                "stim_shape":modelrdm.stimfeature[:,1],
                                "train_test":modelrdm.stimgroup.flatten(),
                                }).assign(subid=subid)
        mds_df = pd.concat((mds_df,X_df),axis=1)

        return mds_df
    
    def _singleparticipant_ROIPS(self,subid,outputdir):
        ##TODO: it is still stuck if try to run PS with four sessions

        # get activity pattern matrix
        X, _, _ = self.get_neuralRDM(subid) #  X, _, _ = self.get_neuralRDM(subid,preproc="PCA")
        # get stimuli properties
        modelrdm  = self.get_modelRDM(subid)
        stim_dict = {"stimid":modelrdm.stimid.flatten(), 
                     "stimsession":modelrdm.stimsession.flatten(), 
                     "stimloc":modelrdm.stimloc, "stimfeature":modelrdm.stimfeature, 
                     "stimgroup":modelrdm.stimgroup.flatten()}        
        
        PS_estimator = NeuralDirectionCosineSimilarity(activitypattern = X,stim_dict=stim_dict)
        PS_estimator.fit()
        #PS_estimator.dir_pair_df.to_csv(os.path.join(outputdir,f'{subid}.csv'))

        return PS_estimator.resultdf.assign(subid=subid)
    
    def _singleparticipant_ROISVMAxis(self,subid,estimator_kwargs):
        X, _, _ = self.get_neuralRDM(subid)
        modelrdm  = self.get_modelRDM(subid)
        stim_dict = {"stimid":modelrdm.stimid.flatten(), 
                     "stimsession":modelrdm.stimsession.flatten(), 
                     "stimloc":modelrdm.stimresploc, "stimfeature":modelrdm.stimfeature, 
                     "stimgroup":modelrdm.stimgroup.flatten()}
        SVM_estimator = SVMDecoder(activitypattern=X,stim_dict=stim_dict,**estimator_kwargs)
         
        SVM_estimator.fit()
        return SVM_estimator.result
    
############################################### GROUP LEVEL  ANALSYSIS METHODS ###################################################   
    def run_ROIRSA(self,njobs:int=1,roirsa_config:dict={}):#cpu_count()
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIRSA)(subid,**roirsa_config) for subid in self.participants)
        corr_df = pd.concat([x[0] for x in dfs_list],axis=0) 
        rdm_df = pd.concat([x[1] for x in dfs_list],axis=0) 
        return corr_df,rdm_df
    
    def run_ROIMDS(self,njobs:int=1,mds_config={}):#cpu_count()
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIMDS)(subid,**mds_config) for subid in self.participants)
        mds_df = pd.concat(dfs_list,axis=0) 
        return mds_df
    
    def run_SearchLightRSA(self,radius:float,outputdir:str,njobs:int=cpu_count()-1,
                           analyses:dict={}):
        """running searchlight analysis

        Parameters
        ----------
        radius : float
            the radius of searchlight
        outputdir : str
            output directory of searchlight analysis
        analyses : list
            list of dictionaries. each dict represent the arguments passed to a type of analysis
        njobs : int, optional
            number of parallel jobs, by default cpu_count()-1
        """
        sphere_vox_count = []
        for j,subid in enumerate(self.participants):
            print(f'running searchlight in {j}/{len(self.participants)}: {subid}')
            ## get neural data
            beta_imgs,vs_masks,proc_masks = self.get_imagedir(subid)
            mask_imgs = vs_masks + self.anatmasks

            subRSA = RSASearchLight(
                        patternimg_paths = beta_imgs,
                        mask_img_path    = mask_imgs,
                        process_mask_img_path = proc_masks,
                        radius=radius,
                        njobs=njobs
                        )
            sphere_vox_count.append(
                np.array(subRSA.A.sum(axis=1)).squeeze()
                )

            ## compute model rdm
            modelrdm  = self.get_modelRDM(subid)
            
            # run search light
            for A in analyses:
                if A["type"] == "decoding":
                    print('running classification decoding analysis searchlight')
                    stim_dict = {"stimid":modelrdm.stimid.flatten(), 
                                 "stimsession":modelrdm.stimsession.flatten(), 
                                 "stimloc":modelrdm.stimloc,
                                 "stimfeature":modelrdm.stimfeature, 
                                 "stimgroup":modelrdm.stimgroup.flatten()}        
                    subRSA.run(
                        estimator = SVMDecoder,
                        estimator_kwargs = {"stim_dict":stim_dict,"categorical_decoder":True,"seed":None,"PCA_component":None},
                        outputpath   = os.path.join(outputdir,'decoding_AxisLocDiscrete','first',subid), 
                        outputregexp = 'acc_%04d.nii', 
                        verbose      = j == 0
                        )# only show details at the first participant
                    print('running regression decoding analysis searchlight')
                    subRSA.run(
                        estimator = SVMDecoder,
                        estimator_kwargs = {"stim_dict":stim_dict,"categorical_decoder":False,"seed":None,"PCA_component":None},
                        outputpath   = os.path.join(outputdir,'decoding_AxisLocContinuous','first',subid), 
                        outputregexp = 'rsquare_%04d.nii', 
                        verbose      = j == 0
                        )# only show details at the first participant
                    
                elif A["type"] == "regression":                
                    print(f'running regression searchlight {A["name"]} - all')                    
                    m_regs = A["regressors"]
                    regress_models = [modelrdm.models[m] for m in m_regs]
                    subRSA.run(
                        estimator = MultipleRDMRegression,
                        estimator_kwargs = {"modelrdms":regress_models, "modelnames":m_regs, "standardize":True},
                        outputpath   = os.path.join(outputdir,"regression",A["name"],'first',subid), 
                        outputregexp = 'beta_%04d.nii', 
                        verbose      = j == 0
                        ) # only show details at the first participant
                    
                    if self.nsession>1: # if multiple runs do between and within run as well
                        print(f'running regression searchlight {A["name"]} - betweenrun')
                        m_regs = [f'between_{x}' for x in A["regressors"]]
                        regress_models = [modelrdm.models[m] for m in m_regs]
                        subRSA.run(
                            estimator = MultipleRDMRegression,
                            estimator_kwargs = {"modelrdms":regress_models, "modelnames":m_regs, "standardize":True},
                            outputpath   = os.path.join(outputdir,"regression",f'{A["name"]}_between','first',subid), 
                            outputregexp = 'beta_%04d.nii', 
                            verbose      = j == 0
                            ) # only show details at the first participant
                        
                        print(f'running regression searchlight {A["name"]} - withinrun')
                        m_regs = [f'within_{x}' for x in A["regressors"]]
                        regress_models = [modelrdm.models[m] for m in m_regs]
                        subRSA.run(
                            estimator = MultipleRDMRegression,
                            estimator_kwargs = {"modelrdms":regress_models, "modelnames":m_regs, "standardize":True},
                            outputpath   = os.path.join(outputdir,"regression",f'{A["name"]}_within','first',subid), 
                            outputregexp = 'beta_%04d.nii', 
                            verbose      = j == 0
                            ) # only show details at the first participant
            
                elif A["type"] == "correlation":
                    if "modelrdms" in A.keys():
                        corr_rdm_names = A["modelrdms"]
                    else:
                        corr_rdm_names = [x for x in modelrdm.models.keys() if not np.logical_or(x.endswith('session'),x.endswith('stimuli'))]
                        if self.taskname == "localizer":
                            corr_rdm_names = [x for x in corr_rdm_names if not np.logical_or(x.endswith('session'),x.endswith('stimuligroup'))]
                    corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]

                    print('running correlation searchlight')
                    subRSA.run(
                        estimator = PatternCorrelation,
                        estimator_kwargs = {"modelrdms":corr_rdm_vals, "modelnames":corr_rdm_names, "type":"spearman"},
                        outputpath   = os.path.join(outputdir,'correlation',A["name"],'first',subid), 
                        outputregexp = 'rho_%04d.nii', 
                        verbose      = j == 0
                        )# only show details at the first participant

                elif A["type"] == "neuralvector":
                    print('running neuralvector analysis searchlight')
                    stim_dict = {"stimid":modelrdm.stimid.flatten(), "stimsession":modelrdm.stimsession.flatten(), "stimloc":modelrdm.stimloc, "stimfeature":modelrdm.stimfeature, "stimgroup":modelrdm.stimgroup.flatten()}        
                    subRSA.run(
                        estimator = NeuralDirectionCosineSimilarity,
                        estimator_kwargs = {"stim_dict":stim_dict,"seed":None},
                        outputpath   = os.path.join(outputdir,'cosinesimilarity',A["name"],'first',subid), 
                        outputregexp = 'ps_%04d.nii', 
                        verbose      = j == 0
                        )# only show details at the first participant

        dump(sphere_vox_count,os.path.join(outputdir,'searchlight_voxcount.pkl'))

    def run_randomROIRSA(self,n_permutations:int=5000,njobs:int=cpu_count()):
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                    delayed(self._singleparticipant_randomROIRSA)(
                    subid,verbose=True,n_permutations=n_permutations
                    ) for subid in self.participants)
            perms_df = pd.concat(dfs_list,axis=0)
            nullgroupmu_df = perms_df.groupby(["analysis","randomseed"]).mean(numeric_only=True).reset_index()
        return perms_df,nullgroupmu_df
    
    def run_ROIPS(self,outputdir:str,njobs:int=1):
        checkdir(outputdir)
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIPS)(subid,outputdir) for subid in self.participants)
        PS_df = pd.concat(dfs_list,axis=0) 
        return PS_df