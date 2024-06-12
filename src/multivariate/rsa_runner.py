"""
This module contains class that wraps up RSA analysis in ROI or whole brain into a pipeline.
The runner class is dependent on the file structure as specified in `FILESTRUCTURE.md`

Zilu Liang @HIPlab Oxford
2023
"""

import itertools
import warnings
                    
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
import nibabel as nib
from nilearn.masking import apply_mask

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import compute_rdm, compute_rdm_nomial, compute_rdm_identity
from zpyhelper.MVPA.estimators import PatternCorrelation,MultipleRDMRegression,NeuralRDMStability
from zpyhelper.MVPA.preprocessors import chain_steps,scale_feature, average_odd_even_session, average_flexi_session,normalise_multivariate_noise,extract_pc
from zpyhelper.image.niidatahandler import retrieve_data_from_image
from zpyhelper.image.searchlight import MVPASearchLight

from multivariate.MVPA_estimator import CompositionalRSA

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'src'))
from multivariate.modelrdms import ModelRDM


PREPROC_CATELOGUE = {
    "MVNN":normalise_multivariate_noise,
    "AOE":average_odd_even_session,
    "ATOE":average_flexi_session,
    "Scale":scale_feature,
    "PCA":extract_pc
}

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
    res_dir : list
        the parent directory that holds participants' folder of residual images that are used to construct the variance-covariance matrix for whitening
    res_fname : list
        _the name of the residual image in each participant's folder
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
                 ## directories
                 beta_dir:list, beta_fname:list,
                 vsmask_dir:list,vsmask_fname:list,
                 pmask_dir:list=None,pmask_fname:list=None,
                 res_dir:list=None,  res_fname:list=None,
                 anatmasks:list=None,
                 taskname:str="localizer",
                 # configurations
                 config_modelrdm:dict={"nan_identity":True,"splitgroup":False},
                 config_neuralrdm:dict={"preproc":None,"distance_metric":"correlation"}
                 ) -> None:
        """
        set up the paths to the image and set up other configurations
        """	
        
        self.participants = participants # participant list
        self.fmribeh_dir  = fmribeh_dir  # behavioral data directory to look for stimuli information

        # diretory and file names of activity pattern image and residual image:
        self.beta_dir     = beta_dir
        self.beta_fname   = beta_fname
        self.res_dir      = res_dir
        self.res_fname    = res_fname
        # diretory and file names of voxel selection masks: participant specific masks of voxels included in computing neural rdm
        self.vsmask_dir   = vsmask_dir
        self.vsmask_fname = vsmask_fname
        # diretory and file names of process mask:  participant-specific masks of voxels used to generate searchlight spheres
        self.pmask_dir    = vsmask_dir if pmask_dir is None else pmask_dir
        self.pmask_fname  = vsmask_fname if pmask_fname is None else pmask_fname
        self.anatmasks    = anatmasks  # anatomical masks: non-participant-specific masks

        # task
        assert taskname in ["localizer", "navigation","both"], "invalid task name!"
        self.taskname = taskname
        
        # options
        self.config_modelrdm  = config_modelrdm
        self.config_neuralrdm = config_neuralrdm
        
        # validate preproc methods
        valid_preproc = list(PREPROC_CATELOGUE.keys())
        spec_steps = list(self.config_neuralrdm["preproc"].keys())
        assert all([x in valid_preproc for x in spec_steps]), f"{np.array(spec_steps)[[x not in valid_preproc for x in spec_steps]]} not a valid preproc step. Valid steps are {valid_preproc}"
        
        # step up proper preproc parameters
        n_stim       = {"navigation":25,"localizer":9}
        n_scanperrun = {"navigation":296,"localizer":326}
        n_run        = {"navigation":4,"localizer":1}
        if "MVNN" in spec_steps:
            self.config_neuralrdm["preproc"]["MVNN"][0] = PREPROC_CATELOGUE["MVNN"]
            if taskname in n_run.keys():
                self.config_neuralrdm["preproc"]["MVNN"][1] = {
                        "ap_groups":    np.concatenate([np.ones((n_stim[taskname],))*j for j in range(n_run[taskname])]),
                        "resid_groups": np.concatenate([np.ones((n_scanperrun[taskname],))*j for j in range(n_run[taskname])])
                    }
            else:
                assert taskname == "both"
                task_seq = ["navigation"]*n_run["navigation"] + ["localizer"]*n_run["localizer"] 
                self.config_neuralrdm["preproc"]["MVNN"][1] = {
                    "ap_groups":    np.concatenate([np.ones((n_stim[t],))*j for j,t in enumerate(task_seq)]),
                    "resid_groups": np.concatenate([np.ones((n_scanperrun[t],))*j for j,t in enumerate(task_seq)])
                }
            
        if "AOE" in spec_steps:
            if taskname == "localizer":
                self.config_neuralrdm["preproc"].pop("AOE")
            if taskname == "navigation":
                self.config_neuralrdm["preproc"]["AOE"][0] = PREPROC_CATELOGUE["AOE"]
                self.config_neuralrdm["preproc"]["AOE"][1] = {
                    "session":np.concatenate([np.ones((n_stim[taskname],))*j for j in range(n_run[taskname])])
                }
        
        if "ATOE" in spec_steps:
            assert taskname == "both"
            task_seq = ["navigation"]*n_run["navigation"] + ["localizer"]*n_run["localizer"]
            self.config_neuralrdm["preproc"]["ATOE"][0] = PREPROC_CATELOGUE["ATOE"]
            self.config_neuralrdm["preproc"]["ATOE"][1] = {
                "session":np.concatenate([np.ones((n_stim[t],))*j for j,t in enumerate(task_seq)]),
                "average_by":[[0,2],[1,3],[4]]
            }

        # define number of sessions based on preproc options        
        if taskname == "both":
            if "ATOE" in self.config_neuralrdm["preproc"].keys(): 
                self.nsession = [2,1]
            else:
                self.nsession = [4,1]
        else:
            self.nsession = n_run[taskname]
            if taskname == "navigation" and "AOE" in self.config_neuralrdm["preproc"].keys():           
                    self.nsession = 2


############################################### PARTICIPANT-SPECIFIC DATA EXTRACTION METHODS ###################################################
    def get_imagedir(self,subid)->tuple:
        """construct the directory to different images for each participants

        Parameters
        ----------
        subid : str
            participants id

        Returns
        -------
        tuple 
            a tuple of paths to different images: beta_imgs,mask_imgs,pmask_imgs,res_imgs
        """
        beta_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.beta_dir,self.beta_fname)]
        mask_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.vsmask_dir,self.vsmask_fname)]
        pmask_imgs = [os.path.join(d,'first',subid,f) for d,f in zip(self.pmask_dir,self.pmask_fname)]
        ## if multivariate noise normalisation is required, we will need to specify the residual image paths
        ## to estimate the voxel-voxel variance-covariance matrix
        if "MVNN" in self.config_neuralrdm["preproc"].keys():
            res_imgs   = [os.path.join(d,'first',subid,f) for d,f in zip(self.res_dir,self.res_fname)]
        else:
            res_imgs = []
        return beta_imgs,mask_imgs,pmask_imgs,res_imgs
    
    def _get_stimbehav_singletask(self,subid:str,taskname:str,nsession:int)->tuple:
        """retrieve behavioural data and stimuli information from a single task for a participants
        Parameters
        ----------
        subid : str
            participant id
        taskname: str
            the task data that is being retrieved
        nsession: int
            the number of sessions of the task. 
            Note that this is not the actual number of runs that the task is scanned. It is the number of session remains after preprocessing the activity pattern matrix.
            For example, the navigation task has 4 runs, but if odd runs and even runs are averaged separately during preprocessing, then ``nsession=2``

        Returns
        -------
        tuple
            a tuple of stimid, stimgtloc, stimfeature, stimgroup, stimresploc, sessions
        """
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

        filter = np.where(stim_train==1) if taskname == "localizer" else (np.arange(stim_train.size),)
        stimid,stimgtloc,stimfeature,stimgroup =  stim_id[filter],stim_loc[filter],stim_feature[filter],stim_train[filter]    
        nstim_single_sess = stimid.size
        
        # copy over to n sessions
        stimid      = np.tile(np.atleast_2d(stimid).T,(nsession,1))
        stimgtloc   = np.tile(stimgtloc,(nsession,1))
        stimfeature = np.tile(stimfeature,(nsession,1))
        stimgroup   = np.tile(np.atleast_2d(stimgroup).T,(nsession,1))

        # load behavioral response
        participant_resp_fns = glob.glob(os.path.join(fmribeh_dir,subid,'sub*_task-piratenavigation_run-*.csv'))
        resp_lists = [pd.read_csv(x, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False).iloc[filter[0],:] for x in participant_resp_fns]
        if taskname == "navigation":
            if nsession == 4:
                resp_df_runs = pd.concat(resp_lists,axis=0)
                stimresploc_ori = np.array(resp_df_runs[['resp_x','resp_y']])
            elif nsession ==1:
                stimresploc_ori = np.mean([np.array(x[['resp_x','resp_y']]) for x in resp_lists],axis=0)
            elif nsession ==2:
                stimresploc_odd  = np.mean([np.array(x[['resp_x','resp_y']]) for x in [resp_lists[0],resp_lists[2]]],axis=0)
                stimresploc_even = np.mean([np.array(x[['resp_x','resp_y']]) for x in [resp_lists[1],resp_lists[3]]],axis=0)
                stimresploc_ori = np.concatenate([stimresploc_odd,stimresploc_even],axis=0)
        else:
            stimresploc_ori = np.mean([np.array(x[['resp_x','resp_y']]) for x in resp_lists],axis=0)
        # rescale response location the same way as stimuli location
        stimresploc = (new_range*(stimresploc_ori-np.min(stim_locori))/old_range) - 1

        #session
        sessions = np.vstack([np.ones((nstim_single_sess,1))*x for x in range(nsession)])

        return stimid, stimgtloc, stimfeature, stimgroup, stimresploc, sessions

    def get_stimbehav(self,subid:str,taskname:str=None):
        """retrieve behavioural data and stimuli information for a participants
        Parameters
        ----------
        subid : str
            participant id

        Returns
        -------
        tuple
            a tuple of stimid, stimgtloc, stimfeature, stimgroup, stimresploc, sessions
        """
        taskname = self.taskname if taskname is None else taskname
        assert taskname in ["navigation","localizer","both"], "Invalid taskname!"
        if taskname == "both":
            outputs = [self._get_stimbehav_singletask(subid, taskname=x, nsession=s) for x,s in zip(["navigation","localizer"],self.nsession)]
            [stimid, stimgtloc, stimfeature, stimgroup, stimresploc] = [np.concatenate([op[k] for op in outputs],axis=0) for k in range(5)] 
            sessions = np.vstack([outputs[0][5],outputs[1][5]+self.nsession[0]]) # n session of navigation + n session of localizer
        else:
            stimid, stimgtloc, stimfeature, stimgroup, stimresploc, sessions = self._get_stimbehav_singletask(subid, taskname=taskname, nsession=self.nsession)

        return stimid, stimgtloc, stimfeature, stimgroup, stimresploc, sessions
        
    def get_modelRDM(self,subid)->ModelRDM:
        """compute model RDM of a participant based on their data

        Parameters
        ----------
        subid : str
            participant id

        Returns
        -------
        an `ModelRDM` object
            `ModelRDM` object
        """
        stimid, stimgtloc, stimfeature, stimgroup, stimresploc, sessions = self.get_stimbehav(subid)        
        
        return ModelRDM(stimid      = stimid,
                        stimgtloc   = stimgtloc,
                        stimfeature = stimfeature,
                        stimgroup   = stimgroup,
                        stimresploc = stimresploc,
                        sessions   = sessions,
                        **self.config_modelrdm)

    def get_neuralBetaResidual(self,subid)->tuple:
        """ extract activty pattern matrix and residual matrix without preprocessing

        Parameters
        ----------
        subid : str
            participant id

        Returns
        -------
        tuple
            a tuple of output, including: activity pattern matrix, residual data matrix
        """
        beta_imgs,vs_masks,_,res_imgs = self.get_imagedir(subid)
        mask_imgs = vs_masks + self.anatmasks
        activitypattern,mask = retrieve_data_from_image(beta_imgs,mask_imgs,returnmask=True)
        if len(res_imgs)>0:
            res_list = [apply_mask(x, mask,ensure_finite = False) for x in res_imgs]
            res_data = np.concatenate(res_list,axis=0)
        else:
            res_data = np.array([])
        return activitypattern, res_data


    def get_neuralRDM(self,subid)->tuple:
        """ extract activty pattern matrix, (pre-)process pattern matrix and generate neural RDM

        Parameters
        ----------
        subid : str
            participant id

        Returns
        -------
        tuple
            a tuple of output, including: (pre-)processed activity pattern matrix, neural RDM, un(pre-)processed activity pattern matrix
        """

        ## get neural data
        X,R = self.get_neuralBetaResidual(subid)
        
        ## preprocessing
        preproc = deepcopy(self.config_neuralrdm["preproc"])
        if "MVNN" in preproc.keys():
            preproc["MVNN"][1]["residualmatrix"] = R
        preproc_func = chain_steps(*list(preproc.values()))
        activitypattern = preproc_func(X)
        return activitypattern, compute_rdm(activitypattern,self.config_neuralrdm["distance_metric"]), X
        
############################################### SINGLE PARTICIPANT LEVEL ANALSYSIS METHODS ###################################################
    def _singleparticipant_ROIRSA(self, subid,
               analyses:list=[],
               returnX=False,
               verbose:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## compute model rdm
        modelrdm  = self.get_modelRDM(subid)        

        ## get neural data
        activitypattern, neural_rdm, _ = self.get_neuralRDM(subid)
        neuralrdmdf = modelrdm.rdm_to_df(modelnames="neural",rdms=neural_rdm)
        neuralrdmdf = neuralrdmdf.assign(subid=subid)

        nsession = np.sum(self.nsession)

        res_df_list = []
        for A in analyses:
            if A["type"] == "regression":                
                m_regs = A["regressors"]
                if nsession>1: # if multiple runs do between and within run as well
                    if verbose:
                        sys.stderr.write(f'running regression in ROI {A["name"]} - betweenrun\r')
                    m_regs = [f'between_{x}' for x in A["regressors"]]
                    if not all([x in modelrdm.models.keys() for x in m_regs]):
                        warnings.warn("this analysis contains invalid regressor so it will be skipped")
                    else:
                        regress_models = [modelrdm.models[m] for m in m_regs]
                        reg_estimator = MultipleRDMRegression(activitypattern,
                                                modelrdms=regress_models,
                                                modelnames=m_regs,
                                                standardize=True,
                                                rdm_metric=self.config_neuralrdm["distance_metric"])
                        reg_estimator.fit()
                        mrdm_name_dict = dict(zip(range(len(m_regs)),m_regs))
                        reg_df = pd.DataFrame(reg_estimator.result).T.rename(columns = mrdm_name_dict).assign(analysis=A["name"],subid=subid)
                        res_df_list.append(reg_df)
                else:
                    if verbose:
                        sys.stderr.write(f'running regression in ROI {A["name"]} - all\r')                                        
                    if not all([x in modelrdm.models.keys() for x in m_regs]):
                        warnings.warn("this analysis contains invalid regressor so it will be skipped")
                    else:
                        regress_models = [modelrdm.models[m] for m in m_regs]
                        reg_estimator = MultipleRDMRegression(activitypattern,
                                                    modelrdms=regress_models,
                                                    modelnames=m_regs,
                                                    standardize=True,
                                                    rdm_metric=self.config_neuralrdm["distance_metric"])
                        reg_estimator.fit()
                        regmrdm_name_dict = dict(zip(range(len(m_regs)),m_regs))
                        reg_df = pd.DataFrame(reg_estimator.result).T.rename(columns = regmrdm_name_dict).assign(analysis=A["name"],subid=subid)
                        res_df_list.append(reg_df)
                    
            elif A["type"] == "correlation":
                assert "modelrdms" in A.keys(), "must specify the model rdms to run correlation with!"
                corr_rdm_names = deepcopy(A["modelrdms"])
                
                if nsession>1: # if multiple runs do between and within run as well
                    corr_rdm_names = corr_rdm_names + [f'between_{x}' for x in A["modelrdms"] if x in modelrdm.models.keys()] + [f'within_{x}' for x in A["modelrdms"] if x in modelrdm.models.keys()]
                
                corr_rdm_names = [x for x in A["modelrdms"] if x in modelrdm.models.keys()]

                corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]

                # run analysis
                if verbose:
                    sys.stderr.write(f"running correlation in ROI {A['name']}\r")
                corr_estimator = PatternCorrelation(activitypattern,
                                   modelrdms=corr_rdm_vals, 
                                   modelnames=corr_rdm_names,
                                   type="spearman",
                                   rdm_metric=self.config_neuralrdm["distance_metric"])
                corr_estimator.fit()
                mrdm_name_dict = dict(zip(range(len(corr_rdm_names)),corr_rdm_names))
                corr_df = pd.DataFrame(corr_estimator.result).T.rename(columns = mrdm_name_dict).assign(analysis=A["name"],subid=subid)
                res_df_list.append(corr_df)
                
            elif A["type"] == "representation_stability":
                groups = np.concatenate([np.ones((25,))*j for j in range(nsession)])

                # run analysis
                if verbose:
                    sys.stderr.write(f"running correlation in ROI {A['name']}")
                ns_estimator = NeuralRDMStability(activitypattern,
                                   groups=groups,
                                   type="spearman")
                ns_estimator.fit()
                ns_df = pd.DataFrame(ns_estimator.result,index=[0]).T.rename(columns = {0:"neuralstability"}).assign(analysis=A["name"],subid=subid)
                res_df_list.append(ns_df)

        if returnX:
            return pd.concat(res_df_list,axis=0).reset_index(drop=True),neuralrdmdf, activitypattern    
        else:
            return pd.concat(res_df_list,axis=0).reset_index(drop=True),neuralrdmdf
        
############################################### RUN SINGLE PARTICIPANT LEVEL ANALYSIS IN BATCH ###################################################   
    def run_ROIRSA(self,njobs:int=1,corr_rdm_names=None,analyses=[],returnX=False,verbose=False):
        with Parallel(n_jobs=njobs) as parallel:
            op_list = parallel(
                delayed(self._singleparticipant_ROIRSA)(subid,corr_rdm_names,analyses,returnX=returnX,verbose=verbose) for subid in self.participants)
        corr_df = pd.concat([x[0] for x in op_list],axis=0) 
        rdm_df = pd.concat([x[1] for x in op_list],axis=0)
        if returnX:
            ROIX = [x[2] for x in op_list]
            return corr_df,rdm_df,ROIX
        else:
            return corr_df,rdm_df   

    def run_SearchLight(self,radius:float,outputdir:str,njobs:int=cpu_count()-1,
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
            print(f'running searchlight in {j+1}/{len(self.participants)}: {subid}')
            
            ## get neural data path 
            beta_imgs,vs_masks,proc_masks,res_imgs = self.get_imagedir(subid)
            mask_imgs = vs_masks + self.anatmasks
            proc_masks = proc_masks + self.anatmasks

            ## instantiate a SearchLight class, searchlight spheres will also be generated at this step
            subSearchLight = MVPASearchLight(
                        patternimg_paths = beta_imgs,
                        mask_img_path    = mask_imgs,
                        residimg_paths   = res_imgs,
                        process_mask_img_path = proc_masks,
                        radius=radius,
                        preproc_steps=self.config_neuralrdm["preproc"],
                        njobs=njobs
                        )
            sphere_vox_count.append(
                np.array(subSearchLight.A.sum(axis=1)).squeeze()
                )

            ## compute model rdm
            modelrdm  = self.get_modelRDM(subid)
            _,stimgtloc,stimfeature,stimgroup,_,sessions = self.get_stimbehav(subid)
            nsession = np.sum(self.nsession)

            ## loop over searchlight analyses
            for A in analyses:
                if A["type"] == "regression":                
                    m_regs = A["regressors"]

                    print(f'running regression searchlight {A["name"]}') 
                    if nsession>1: # if multiple runs do between run                        
                        m_regs = [f'between_{x}' for x in A["regressors"]]
                        if not all([x in modelrdm.models.keys() for x in m_regs]):
                            warnings.warn("this analysis contains invalid regressor so it will be skipped")
                        else:
                            regress_models = [modelrdm.models[m] for m in m_regs]
                            subSearchLight.run(
                                estimator = MultipleRDMRegression,
                                estimator_kwargs = {"modelrdms":regress_models, "modelnames":m_regs, 
                                                    "standardize":True,"rdm_metric":self.config_neuralrdm["distance_metric"]},
                                outputpath   = os.path.join(outputdir,"regression",f'{A["name"]}_between','first',subid), 
                                outputregexp = 'beta_%04d.nii', 
                                verbose      = j == 0
                                ) # only show details at the first participant
                    else: 
                        if not all([x in modelrdm.models.keys() for x in m_regs]):
                            warnings.warn("this analysis contains invalid regressor so it will be skipped")
                        else:
                            regress_models = [modelrdm.models[m] for m in m_regs]
                            subSearchLight.run(
                                estimator = MultipleRDMRegression,
                                estimator_kwargs = {"modelrdms":regress_models, "modelnames":m_regs,
                                                    "standardize":True,"rdm_metric":self.config_neuralrdm["distance_metric"]},
                                outputpath   = os.path.join(outputdir,"regression",A["name"],'first',subid), 
                                outputregexp = 'beta_%04d.nii', 
                                verbose      = j == 0
                                ) # only show details at the first participant
                        
                elif A["type"] == "correlation":
                    assert "modelrdms" in A.keys(), "must specify the model rdms to run correlation with!"
                    corr_rdm_names = deepcopy(A["modelrdms"])
                    
                    if nsession>1: # if multiple runs do between and within run as well
                        corr_rdm_names = corr_rdm_names + [f'between_{x}' for x in A["modelrdms"] if x in modelrdm.models.keys()] + [f'within_{x}' for x in A["modelrdms"] if x in modelrdm.models.keys()]
                    
                    corr_rdm_names = [x for x in A["modelrdms"] if x in modelrdm.models.keys()]

                    corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]

                    # run analysis
                    print(f"running correlation searchlight {A['name']}")
                    subSearchLight.run(
                        estimator = PatternCorrelation,
                        estimator_kwargs = {"modelrdms":corr_rdm_vals, "modelnames":corr_rdm_names,
                                            "type":"spearman","rdm_metric":self.config_neuralrdm["distance_metric"]},
                        outputpath   = os.path.join(outputdir,'correlation',A["name"],'first',subid), 
                        outputregexp = 'rho_%04d.nii', 
                        verbose      = j == 0
                        )# only show details at the first participant
                
                elif A["type"] == "representation_stability":
                    # run analysis
                    print(f"running correlation searchlight {A['name']}")
                    subSearchLight.run(
                        estimator = NeuralRDMStability,
                        estimator_kwargs = {"groups":sessions, "type":"spearman"},
                        outputpath   = os.path.join(outputdir,'correlation',A["name"],'first',subid), 
                        outputregexp = 'rho_%04d.nii', 
                        verbose      = j == 0
                        )# only show details at the first participant
                
                elif A["type"] == "location_composition":
                    if self.taskname=="both":
                        print(f"running correlation searchlight {A['name']}")
                        task_splitter = (sessions==np.max(sessions)).flatten()
                        control_feature_rdm = compute_rdm_nomial(stimfeature[~task_splitter,:])
                        control_session_rdm = compute_rdm_identity(sessions[~task_splitter,:])
                        loc2stim_resid_kwarg = {
                            "source_ref_split": task_splitter*1,
                            "compose_features":stimgtloc,
                            "compose_feature_names":["x","y"],
                            "control_features":[],
                            "control_feature_names":None,
                            "controlrdms_src":[control_session_rdm,control_feature_rdm],
                            "controlrdms_src_name":["session","feature"],
                            "controlrdms_ref":[control_session_rdm,control_feature_rdm],
                            "session":sessions.flatten()[~task_splitter],
                            "stimgroup":stimgroup.flatten()[~task_splitter]
                        }
                        subSearchLight.run(
                            estimator = CompositionalRSA,
                            estimator_kwargs = loc2stim_resid_kwarg,
                            outputpath   = os.path.join(outputdir,'composition',A["name"],'first',subid), 
                            outputregexp = 'rho_%04d.nii', 
                            verbose      = j == 0
                            )
                    
                elif A["type"] == "feature_composition":
                    if self.taskname=="navigation":
                        print(f"running correlation searchlight {A['name']}")
                        task_splitter = (stimgroup==1).flatten()
                        control_feature_rdm = compute_rdm_nomial(stimfeature[~task_splitter,:])
                        control_session_rdm = compute_rdm_identity(sessions[~task_splitter,:])
                        train2test_resid_kwarg = {
                            "source_ref_split": task_splitter*1,
                            "compose_features":stimgtloc,
                            "compose_feature_names":["x","y"],
                            "control_features":[],
                            "control_feature_names":None,
                            "controlrdms_src":[control_session_rdm,control_feature_rdm],
                            "controlrdms_src_name":["session","feature"],
                            "controlrdms_ref":[control_session_rdm,control_feature_rdm],
                            "session":sessions.flatten()[~task_splitter],
                            "stimgroup":stimgroup.flatten()[~task_splitter]
                        }
                        subSearchLight.run(
                            estimator = CompositionalRSA,
                            estimator_kwargs = train2test_resid_kwarg,
                            outputpath   = os.path.join(outputdir,'composition',A["name"],'first',subid), 
                            outputregexp = 'rho_%04d.nii', 
                            verbose      = j == 0
                            )

        dump(sphere_vox_count,os.path.join(outputdir,'searchlight_voxcount.pkl'))

    