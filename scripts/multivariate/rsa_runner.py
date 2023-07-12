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
from multivariate.helper import ModelRDM, compute_rdm, checkdir, scale_feature
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression

class RSARunner:
    def __init__(self,participants,fmribeh_dir,
                 nsession,
                 beta_dir,beta_fname,
                 vsmask_dir,vsmask_fname,
                 pmask_dir=None,pmask_fname=None,
                 anatmasks=None,
                 taskname:str="localizer") -> None:
        self.participants = participants # participant list
        self.fmribeh_dir  = fmribeh_dir  # behavioral data directory to look for stimuli information

        # diretory and file names of activity pattern image:
        self.beta_dir     = beta_dir
        self.beta_fname   = beta_fname
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

############################################### PARTICIPANT-SPECIFIC DATA EXTRACTION METHODS ###################################################
    def get_imagedir(self,subid):
        beta_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.beta_dir,self.beta_fname)]
        mask_imgs  = [os.path.join(d,'first',subid,f) for d,f in zip(self.vsmask_dir,self.vsmask_fname)]
        pmask_imgs = [os.path.join(d,'first',subid,f) for d,f in zip(self.pmask_dir,self.pmask_fname)]
        return beta_imgs,mask_imgs,pmask_imgs
    
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
        _, stim_color = np.unique(stim_color, return_inverse=True)
        _, stim_shape = np.unique(stim_shape, return_inverse=True)

        stimid = stim_id
        stimloc = np.vstack([stim_locx,stim_locy]).T
        stimfeature = np.vstack([stim_color,stim_shape]).T
        stimgroup = stim_train

        if self.taskname == "localizer":
            training_filter = np.where(stim_train==1)
            return stimid[training_filter],stimloc[training_filter],stimfeature[training_filter],stimgroup[training_filter]
        elif self.taskname == "navigation":
            return stimid,stimloc,stimfeature,stimgroup

    def get_modelRDM(self,subid,randomseed:int=1,nan_identity:bool=True):
        stimid,stimloc,stimfeature,stimgroup = self.load_stimlist(subid)
        return ModelRDM(stimid = stimid,
                        stimloc = stimloc,
                        stimfeature = stimfeature,
                        stimgroup   = stimgroup,
                        n_session   = self.nsession,
                        randomseed  = randomseed,
                        nan_identity = nan_identity)

    def get_neuralRDM(self,subid,
                      preproc:str=None):
        ## model rdm
        modelrdm = self.get_modelRDM(subid)

        ## get neural data
        beta_imgs,vs_masks,_ = self.get_imagedir(subid)
        mask_imgs = vs_masks + self.anatmasks
        APD = ActivityPatternDataLoader(beta_imgs,mask_imgs)
        
        ## preprocessing
        ev = 1
        if preproc is None:
            activitypattern = APD.X
        elif preproc == "cocktail_blank_removal":
            for k in modelrdm.stimgroup:
                activitypattern[np.where(modelrdm.stimgroup == k),:] = scale_feature(activitypattern[np.where(modelrdm.stimgroup == k),:],1,False)
        elif preproc=="subtract_mean":
            activitypattern = scale_feature(activitypattern,1,False)
        elif preproc=="PCA":
            neuralPCA = PCA(n_components=0.9)
            activitypattern = neuralPCA.fit_transform(APD.X)
            ev = np.sum(neuralPCA.fit(APD.X).explained_variance_ratio_)
            
        return activitypattern,compute_rdm(activitypattern,'correlation'),ev
        
############################################### SINGLE PARTICIPANT LEVEL ANALSYSIS METHODS ###################################################
    def _singleparticipant_ROIRSA(self, subid,
               corr_rdm_names=None,
               corr_type="spearman",
               randomseed:int=1,nan_identity:bool=True,
               verbose:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## compute model rdm
        modelrdm  = self.get_modelRDM(subid,randomseed,nan_identity)
        ### choose which model rdm is used for calculating correlation
        if corr_rdm_names is None:
            corr_rdm_names = [x for x in modelrdm.models.keys() if not np.any([x.endswith('session'),x.endswith('stimuli'),x.endswith('stimuligroup')])]
        else:
            corr_rdm_names = [x for x in modelrdm.models.keys() if x in corr_rdm_names]
        corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]
        ### put model rdm into dataframe
        df_inc_models = [x for x in corr_rdm_names  if not np.logical_or(x.startswith('between_'),x.startswith('within_'))]
        modeldf = modelrdm.rdm_to_df(df_inc_models)

        ## get neural data
        _, neural_rdm, _ = self.get_neuralRDM(subid)
        neuralrdmdf = modelrdm.rdm_to_df(modelnames="neural",rdms=neural_rdm)
        neuralrdmdf = neuralrdmdf.join(modeldf).reset_index().assign(subid=subid)
        
        ## compute correlations between neural rdm and model rdms
        if isinstance(corr_type,str):
            corr_type = [corr_type]
        elif not isinstance(corr_type,list):
            raise ValueError('invalid input type')
        
        corr_df = []
        for cr in corr_type:
            PC_s =  PatternCorrelation(neuralrdm = neural_rdm,
                                    modelrdms = corr_rdm_vals,
                                    modelnames = corr_rdm_names,
                                    type=cr)
            PC_s.fit()
            corr_name_dict = dict(zip(range(len(corr_rdm_vals)),corr_rdm_names))
            cr_df = pd.DataFrame(PC_s.result).T.rename(columns = corr_name_dict).assign(analysis=cr,subid=subid)
            corr_df.append(cr_df)
        return pd.concat(corr_df,axis=0),neuralrdmdf

    def _singleparticipant_randomROIRSA(self, subid,
               verbose:bool=False,n_permutations:int=5000):

        t0 = time.time()

        _, neural_rdm, _ = self.get_neuralRDM(subid)

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
               verbose:bool=False):
        if verbose:
            sys.stderr.write(f"{subid}\r")
        
        ## get neural data
        activitypattern, _, _ = self.get_neuralRDM(subid)
        ## compute MDS
        embedding = MDS(
                n_components=2,
                max_iter=5000,
                n_init=100,
                n_jobs=1,
                normalized_stress=False,
                random_state=randomseed,
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
    
    def _singleparticipant_ROIPS(self,subid,outputdir):
        X, _, _ = self.get_neuralRDM(subid)
        #generate a randomly permutated X (randomly permutated within each row):
        randX = np.empty_like(X)
        for j in range(X.shape[0]):
            randX[j,:] = np.random.permutation(X.shape[1])
       
        # get stimuli properties
        modelrdm  = self.get_modelRDM(subid)

        # compute coding directions between any two stimuli
        iter_pairs = enumerate(zip(
            itertools.product(np.arange(len(X)),np.arange(len(X))),
            itertools.product(X,X),
            itertools.product(randX,randX)
            ))
        dirs_mat = np.empty((len(X),len(X),X.shape[1]))
        dirs_mat_rand = np.empty((len(X),len(X),X.shape[1]))
        cols = ['stim1', 'stim2','run1','run2', #stim id and run id of stimulus 1 (starting stim) and 2 (ending stim)
                'sx','ex', #x location on groundtruth map for starting stim (sx) and ending stim (ex)
                'sy','ey', #y location on groundtruth map for starting stim (sy) and ending stim (ey)
                'sc','ec', #colour for starting stim (sc) and ending stim (ec)
                'ss','es'  #shape for starting stim (ss) and ending stim (es)
                ]
        dir_info_mat =  np.empty((len(X),len(X),len(cols)))
        for jpair,((jpS1,jpS2), (u,v), (urand,vrand)) in list(iter_pairs):
            dirs_mat[jpS1,jpS2,:] = u-v
            dirs_mat_rand[jpS1,jpS2,:] = urand - vrand
            dir_info_mat[jpS1,jpS2,:] = np.array([modelrdm.stimid.flatten()[jpS1],
                                                  modelrdm.stimid.flatten()[jpS2],
                                                  modelrdm.stimsession.flatten()[jpS1],
                                                  modelrdm.stimsession.flatten()[jpS2],
                                                  modelrdm.stimloc[:,0][jpS1],
                                                  modelrdm.stimloc[:,0][jpS2],
                                                  modelrdm.stimloc[:,1][jpS1],
                                                  modelrdm.stimloc[:,1][jpS2],
                                                  modelrdm.stimfeature[:,0][jpS1],
                                                  modelrdm.stimfeature[:,0][jpS2],
                                                  modelrdm.stimfeature[:,1][jpS1],
                                                  modelrdm.stimfeature[:,1][jpS2]
                                                  ])

        
        dp_idx = np.tril_indices(dirs_mat.shape[0], k = -1)
        dirs_arr = dirs_mat[dp_idx]
        dirs_arr_rand = dirs_mat_rand[dp_idx]
        dir_info_arr = dir_info_mat[dp_idx] #dir_info_mat.reshape((-1,len(cols)))
        dir_df = pd.DataFrame(dir_info_arr,columns=cols)
        
        # cosine similarity between pairs of coding directions
        iter_dir_pairs = list(itertools.combinations(dirs_arr,r=2))
        iter_dir_pairs_rand = list(itertools.combinations(dirs_arr_rand,r=2))
        idx_pairs = list(itertools.combinations(np.array(dir_df.index),r=2))                    

        cos_sim  = np.array([1 - scipy.spatial.distance.cosine(dir1,dir2) for dir1,dir2 in iter_dir_pairs])
        cos_sim_rand  = np.array([1 - scipy.spatial.distance.cosine(dir1,dir2) for dir1,dir2 in iter_dir_pairs_rand])

        dir_df = dir_df.assign(tmpvar=1).reset_index() # create a temporary variable so that we can use merge without index
        create_dir_pair_df = lambda idx1,idx2,dir_df: pd.merge(dir_df.loc[[idx1]], dir_df.loc[[idx2]], on="tmpvar", how="outer", suffixes=('_dir1', '_dir2'))
        dir_pair_df = pd.concat([create_dir_pair_df(idx1,idx2,dir_df)  for idx1,idx2 in idx_pairs],axis=0)
        dir_pair_df["neural"] = cos_sim
        dir_pair_df["neuralrand"] = cos_sim_rand

        def get_pair_type_loc(df_row):
            df_row["same_sx"] = df_row["sx_dir1"] == df_row["sx_dir2"]
            df_row["same_sy"] = df_row["sy_dir1"] == df_row["sy_dir2"]
            df_row["same_ex"] = df_row["ex_dir1"] == df_row["ex_dir2"]
            df_row["same_ey"] = df_row["ey_dir1"] == df_row["ey_dir2"]

            ## two coding directions that have the same start and end x but in different y rows
            if df_row['same_sx'] & df_row['same_ex'] & (df_row['sy_dir1'] == df_row['ey_dir1']) & (df_row['sy_dir2'] == df_row['ey_dir2']):
                pair_type = "betweenX"
            
            ## two coding directions that have the same start and end y but in different x columns
            elif df_row['same_sy'] & df_row['same_ey'] & (df_row['sx_dir1'] == df_row['ex_dir1']) & (df_row['sx_dir2'] == df_row['ex_dir2']):
                pair_type = "betweenY"

            ## two coding directions that are in the same x column
            elif np.all([y == df_row['sy_dir1'] for y in [df_row['sy_dir1'],df_row['ey_dir1'],df_row['sy_dir2'],df_row['ey_dir2']]]):
                pair_type = "withinX"

            ## two coding directions that are in the same y row
            elif np.all([x == df_row['sx_dir1'] for x in [df_row['sx_dir1'],df_row['ex_dir1'],df_row['sx_dir2'],df_row['ex_dir2']]]):
                pair_type = "withinY"
            else:
                pair_type = "others"
            return pair_type
        
        def get_pair_type_feature(df_row):
            df_row["same_sc"] = df_row["sc_dir1"] == df_row["sc_dir2"]
            df_row["same_ss"] = df_row["ss_dir1"] == df_row["ss_dir2"]
            df_row["same_ec"] = df_row["ec_dir1"] == df_row["ec_dir2"]
            df_row["same_es"] = df_row["es_dir1"] == df_row["es_dir2"]
            ## two coding directions that have the same start and end colours but in different shape rows
            if df_row['same_sc'] & df_row['same_ec'] & (df_row['ss_dir1'] == df_row['es_dir1']) & (df_row['ss_dir2'] == df_row['es_dir2']):
                pair_type = "betweenColour"
            ## two coding directions that have the same start and end shape but in different colour columns
            elif df_row['same_ss'] & df_row['same_es'] & (df_row['sc_dir1'] == df_row['ec_dir1']) & (df_row['sc_dir2'] == df_row['ec_dir2']):
                pair_type = "betweenShape"
            ## two coding directions that are in the same colour column
            elif np.all([y == df_row['ss_dir1'] for y in [df_row['ss_dir1'],df_row['es_dir1'],df_row['ss_dir2'],df_row['es_dir2']]]):
                pair_type = "withinColour"
            ## two coding directions that are in the same shape row
            elif np.all([x == df_row['sc_dir1'] for x in [df_row['sc_dir1'],df_row['ec_dir1'],df_row['sc_dir2'],df_row['ec_dir2']]]):
                pair_type = "withinShape"
            else:
                pair_type = "others"
            return pair_type

        dir_pair_df = dir_pair_df.assign(subid=subid)
        dir_pair_df.to_csv(os.path.join(outputdir,f'{subid}.csv'))

        dir_pair_dfl,dir_pair_dff = deepcopy(dir_pair_df),deepcopy(dir_pair_df)
        dir_pair_dfl["pairtype"] = dir_pair_df.apply(get_pair_type_loc, axis=1)
        dir_pair_df_suml = dir_pair_dfl.groupby(['subid','pairtype'])[['neural','neuralrand']].mean().reset_index()
        dir_pair_dff["pairtype"] = dir_pair_df.apply(get_pair_type_feature, axis=1)
        dir_pair_df_sumf = dir_pair_dff.groupby(['subid','pairtype'])[['neural','neuralrand']].mean().reset_index()
        dir_pair_df_sum = pd.concat([
            dir_pair_df_suml[dir_pair_df_suml["pairtype"]!="others"],
            dir_pair_df_sumf[dir_pair_df_sumf["pairtype"]!="others"],
        ],axis=0)

        return dir_pair_df_sum
    
############################################### GROUP LEVEL  ANALSYSIS METHODS ###################################################   
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
    
    def run_ROIPS(self,outputdir:str,njobs:int=1):
        checkdir(outputdir)
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIPS)(subid,outputdir) for subid in self.participants)
        PS_df = pd.concat(dfs_list,axis=0) 
        return PS_df