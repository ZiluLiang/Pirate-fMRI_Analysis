import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import glob
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump
from sklearn.manifold import MDS   

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import ModelRDM, compute_rdm, checkdir, scale_feature
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression

class RSARunner:
    def __init__(self,participants,fmribeh_dir,
                 beta_dir,beta_fname,
                 vsmask_dir,vsmask_fname,
                 pmask_dir,pmask_fname,
                 anatmasks,
                 nsession,
                 taskname:str="localizer") -> None:
        self.participants = participants # participant list
        self.fmribeh_dir  = fmribeh_dir  # behavioral data directory

        # diretory and file names of activity pattern image:
        self.beta_dir     = beta_dir
        self.beta_fname   = beta_fname
        # diretory and file names of voxel selection masks: participant specific masks of voxels included in computing neural rdm
        self.vsmask_dir   = vsmask_dir
        self.vsmask_fname = vsmask_fname
        # diretory and file names of process mask:  participant-specific masks of voxels used to generate searchlight spheres
        self.pmask_dir    = pmask_dir
        self.pmask_fname  = pmask_fname
        self.anatmasks    = anatmasks  # anatomical masks: non-participant-specific masks

        # task
        assert taskname in {"localizer", "navigation"}, "invalid task name!"
        self.taskname = taskname
        
        # number of sessions
        self.nsession     = nsession

    def load_stimlist(self,subid):
        fmribeh_dir = self.fmribeh_dir
        # load stimuli list
        stim_list_fn = glob.glob(os.path.join(fmribeh_dir,subid,'sub*_stimlist.txt'))[0]
        stim_list    =  pd.read_csv(stim_list_fn, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False)# use `sort_values` to make sure stim list is in the same order for all participants

        stim_id = np.array(stim_list['stim_id']) # get stimuli id
        stim_image = np.array([x.replace('.png','') for x in stim_list["stim_img"]]) # get stimuli image
        stim_train = np.array(stim_list['training']) # get training/test stimuli classification
        # get 2d location
        stim_locx = np.array(stim_list['stim_x'])
        stim_locy = np.array(stim_list['stim_y'])
        # get visual features
        stim_color = np.array([x.replace('.png','').split('_')[0] for x in stim_list["stim_img"]])
        stim_shape = np.array([x.replace('.png','').split('_')[1] for x in stim_list["stim_img"]])        

        stimid = stim_id
        stimloc = np.vstack([stim_locx,stim_locy]).T
        stimfeature = np.vstack([stim_color,stim_shape]).T
        stimgroup = stim_train

        if self.taskname == "localizer":
            training_filter = np.where(stim_train==1)
            return stimid[training_filter],stimloc[training_filter],stimfeature[training_filter],stimgroup[training_filter]
        elif self.taskname == "navigation":
            return stimid,stimloc,stimfeature,stimgroup

    def get_modelRDM(self,subid,randomseed:int=1):
        stimid,stimloc,stimfeature,stimgroup = self.load_stimlist(subid)
        return ModelRDM(stimid = stimid,
                        stimloc = stimloc,
                        stimfeature = stimfeature,
                        stimgroup   = stimgroup,
                        n_session   = self.nsession,
                        randomseed  = randomseed)

    def get_neuralRDM(self,subid,centering:bool=False,group_centering:bool=False):
        ##
        modelrdm = self.get_modelRDM(subid)
        ## get neural data
        beta_imgs,vs_masks,_ = self.get_imagedir(subid)
        mask_imgs = vs_masks + self.anatmasks
        APD = ActivityPatternDataLoader(beta_imgs,mask_imgs)
        activitypattern = APD.X
        if centering:
            activitypattern = scale_feature(activitypattern,1,False)
        if group_centering:
            for k in modelrdm.stimgroup:
                activitypattern[np.where(modelrdm.stimgroup == k),:] = scale_feature(activitypattern[np.where(modelrdm.stimgroup == k),:],1,False)
        return activitypattern,compute_rdm(activitypattern,'correlation')
        
    def get_imagedir(self,subid):
        beta_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.beta_dir,self.beta_fname)]
        mask_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.vsmask_dir,self.vsmask_fname)]
        pmask_imgs = [os.path.join(d,'first',subid,f) for d,f in zip(self.pmask_dir,self.pmask_fname)]
        return beta_imgs,mask_imgs,pmask_imgs
    
    def run_ROIRSA(self,njobs:int=1):#cpu_count()
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIRSA)(subid) for subid in self.participants)
        corr_df = pd.concat([x[0] for x in dfs_list],axis=0) 
        rdm_df = pd.concat([x[1] for x in dfs_list],axis=0) 
        return corr_df,rdm_df
    
    def run_ROIMDS(self,njobs:int=1):#cpu_count()
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIMDS)(subid) for subid in self.participants)
        mds_df = pd.concat(dfs_list,axis=0) 
        return mds_df
    
    def run_SearchLightRSA(self,radius,outputdir,njobs:int=cpu_count()-1):
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
            if self.taskname == "localizer":
                m_regs = ['feature1dx','feature1dy','loc2d']
                corr_rdm_names = [x for x in modelrdm.models.keys() if not np.logical_or(x.endswith('session'),x.endswith('stimuli'))]
                corr_rdm_names = [x for x in corr_rdm_names if not np.logical_or(x.endswith('session'),x.endswith('stimuligroup'))]
            else:
                m_regs = ['feature1dx','feature1dy','stimuligroup','loc2d']
                corr_rdm_names = [x for x in modelrdm.models.keys() if not np.logical_or(x.endswith('session'),x.endswith('stimuli'))]
            regress_models = [modelrdm.models[m] for m in m_regs]
            corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]
            print('running regression searchlight')
            subRSA.run(MultipleRDMRegression,regress_models,m_regs,os.path.join(outputdir,'regression','first',subid), 'beta_%04d.nii', j == 0) # only show details at the first sub
            print('running correlation searchlight')
            subRSA.run(PatternCorrelation,corr_rdm_vals,corr_rdm_names,os.path.join(outputdir,'correlation','first',subid), 'rho_%04d.nii', j == 0)
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
    
    def _singleparticipant_ROIRSA(self, subid,
               corr_rdm_names=None,
               randomseed:int=1,
               verbose:bool=False,
               centering:bool=False,group_centering:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## compute model rdm
        modelrdm  = self.get_modelRDM(subid)
    
        if corr_rdm_names is None:
            corr_rdm_names = [x for x in modelrdm.models.keys() if not np.any([x.endswith('session'),x.endswith('within_stimuli'),x.endswith('stimuligroup')])]
        else:
            corr_rdm_names = [x for x in modelrdm.models.keys() if x in corr_rdm_names]
    
        if self.nsession == 1:
            corr_rdm_names.remove("stimuli")
        corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]

        df_inc_models = [x for x in corr_rdm_names  if not np.logical_or(x.startswith('between_'),x.startswith('within_'))]
        modeldf = modelrdm.rdm_to_df(df_inc_models)

        ## get neural data
        _, neural_rdm = self.get_neuralRDM(subid,centering,group_centering)
        neuralrdmdf = modelrdm.rdm_to_df(modelnames="neural",rdms=neural_rdm)
        neuralrdmdf = neuralrdmdf.join(modeldf).reset_index().assign(subid=subid)
        
        ## compute correlations between neural rdm and model rdms
        PC_s =  PatternCorrelation(neuralrdm = neural_rdm,
                                    modelrdms = corr_rdm_vals,
                                    modelnames = corr_rdm_names,
                                    type="spearman")
        PC_s.fit()
        PC_k =  PatternCorrelation(neuralrdm = neural_rdm,
                                modelrdms = corr_rdm_vals,
                                modelnames = corr_rdm_names,
                                type="kendall")
        PC_k.fit()
        corr_name_dict = dict(zip(range(len(corr_rdm_vals)),corr_rdm_names))
        corr_df = pd.concat(
            [pd.DataFrame(PC_s.result).T.rename(columns = corr_name_dict).assign(analysis="spearman"),
                pd.DataFrame(PC_k.result).T.rename(columns = corr_name_dict).assign(analysis="kendall")],
                axis=0).assign(subid=subid)
        return corr_df,neuralrdmdf

    def _singleparticipant_randomROIRSA(self, subid,
               verbose:bool=False,n_permutations:int=5000,
               centering:bool=False,group_centering:bool=False):

        t0 = time.time()

        _, neural_rdm = self.get_neuralRDM(subid,centering,group_centering)

        ## compute model rdm
        corr_df_list = []
        for j,randseed in enumerate(np.arange(n_permutations)):
            if verbose:
                sys.stderr.write(f"{subid}:  {j} / {n_permutations}\r")
            modelrdm  = self.get_modelRDM(subid,randomseed = randseed)

            randrdm_names = ["shuffledloc2d", "randfeature2d", "randmatrix",
                            "between_shuffledloc2d", "between_randfeature2d", "between_randmatrix",
                            "within_shuffledloc2d", "within_randfeature2d", "within_randmatrix"]
            corr_rdm_names = [x for x in modelrdm.models.keys() if x in randrdm_names]
            corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]
            
            ## compute correlations between neural rdm and model rdms
            PC_s =  PatternCorrelation(neuralrdm = neural_rdm,
                                       modelrdm = corr_rdm_vals,
                                        type="spearman").fit()
            PC_k =  PatternCorrelation(neuralrdm = neural_rdm,
                                    modelrdm = corr_rdm_vals,
                                    type="kendall").fit()
            corr_name_dict = dict(zip(range(len(corr_rdm_vals)),corr_rdm_names))
            corr_df = pd.concat(
                [pd.DataFrame(PC_s.result).T.rename(columns = corr_name_dict).assign(analysis="spearman"),
                    pd.DataFrame(PC_k.result).T.rename(columns = corr_name_dict).assign(analysis="kendall")],
                    axis=0).assign(subid=subid,randomseed=randseed)
            corr_df_list.append(corr_df)
        print(f'{time.time()-t0} seconds elapsed')
        return pd.concat(corr_df_list,axis=0)
    
    def _singleparticipant_ROIMDS(self, subid,
               randomseed:int=1,
               verbose:bool=False,
               centering:bool=False,group_centering:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## get neural data
        activitypattern, _ = self.get_neuralRDM(subid,centering,group_centering)
        ## compute MDS
        embedding = MDS(
                n_components=2,
                max_iter=50000,
                n_init=100,
                n_jobs=10,
                normalized_stress=False,
                )
        X_transformed = embedding.fit_transform(activitypattern)
        modelrdm  = self.get_modelRDM(subid)
        mds_df = pd.DataFrame({"stim_id":modelrdm.stimid.flatten(),
                                "stim_x":modelrdm.stimloc[:,0],
                                "stim_y":modelrdm.stimloc[:,1],
                                "stim_color":modelrdm.stimfeature[:,0],
                                "stim_shape":modelrdm.stimfeature[:,1],
                                "train_test":modelrdm.stimgroup.flatten(),
                                "x":X_transformed[:,0],
                                "y":X_transformed[:,1]
                                }).assign(subid=subid)

        return mds_df