import itertools
import scipy
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
                      preproc:str=None,
                      PCA:bool=False):
        ## model rdm
        modelrdm = self.get_modelRDM(subid)

        ## get neural data
        beta_imgs,vs_masks,_ = self.get_imagedir(subid)
        mask_imgs = vs_masks + self.anatmasks
        APD = ActivityPatternDataLoader(beta_imgs,mask_imgs)
        
        ## preprocessing
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
            
        return activitypattern,compute_rdm(activitypattern,'correlation')
        
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
        _, neural_rdm = self.get_neuralRDM(subid)
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

        _, neural_rdm = self.get_neuralRDM(subid)

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
        activitypattern, _ = self.get_neuralRDM(subid)
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
    
    def _singleparticipant_ROIPS(self,subid):
        X, _ = self.get_neuralRDM(subid)
        modelrdm  = self.get_modelRDM(subid)
        iter_pairs = enumerate(zip(
            itertools.product(np.arange(len(X)),np.arange(len(X))),
            itertools.product(X,X)
            ))
        dirs_mat = np.empty((len(X),len(X),X.shape[1]))
        cols = ['stim1', 'stim2', 'attrx1', 'attrx2', 'attry1', 'attry2', 'training1',
            'training2','run1','run2']
        dir_info_mat =  np.empty((len(X),len(X),len(cols)),dtype='int')
        for jpair,((jpS1,jpS2), (u,v)) in list(iter_pairs):
            dirs_mat[jpS1,jpS2,:] = u-v
            dir_info_mat[jpS1,jpS2,:] = np.array([modelrdm.stimid.flatten()[jpS1],
                                                  modelrdm.stimid.flatten()[jpS2],
                                                  modelrdm.stimloc[:,0][jpS1],
                                                  modelrdm.stimloc[:,0][jpS2],
                                                  modelrdm.stimloc[:,1][jpS1],
                                                  modelrdm.stimloc[:,1][jpS2],
                                                  modelrdm.stimgroup.flatten()[jpS1],
                                                  modelrdm.stimgroup.flatten()[jpS2],
                                                  modelrdm.stimsession.flatten()[jpS1],
                                                  modelrdm.stimsession.flatten()[jpS2]]).astype('int')

        dirs_arr = dirs_mat.reshape((-1,X.shape[1]))
        dir_info_arr = dir_info_mat.reshape((-1,len(cols)))
        dir_df = pd.DataFrame(dir_info_arr,columns=cols)
        dir_df["same_x"] =  dir_df["attrx1"] ==  dir_df["attrx2"]
        dir_df["same_y"] =  dir_df["attry1"] ==  dir_df["attry2"]
        dir_df["withintrainstim"] =  (dir_df["training1"] ==  1) & (dir_df["training2"] ==  1)
        dir_df["withinrun"] =  dir_df["run1"] ==  dir_df["run2"]
        dir_pairs_dict = {
            'sXdY':dir_df[(dir_df["same_x"])&(~dir_df["same_y"])],
            'dXsY':dir_df[(~dir_df["same_x"])&(dir_df["same_y"])],
            'dXdY':dir_df[(~dir_df["same_x"])&(~dir_df["same_y"])],
        }
        dir_pairs_dict = {
            'dXdY':dir_df[(~dir_df["same_x"])&(~dir_df["same_y"])],
        }
        for x,y in zip(np.arange(5),np.arange(5)):
            dir_pairs_dict |= {
                f'X{x}': dir_df[
                    (dir_df["same_x"]) & (~dir_df["same_y"]) & (dir_df["attrx1"] == x) & (dir_df["attrx2"] == x)
                ],
                f'Y{y}': dir_df[
                    (~dir_df["same_x"]) & (dir_df["same_y"]) & (dir_df["attry1"] == y) & (dir_df["attry2"] == y)
                ],
            }
        def compute_theoretical_dir(sel_pair_df,idx1,idx2):
            # groundtruth feature vectors
            gt_dir1_stim1 = np.array([sel_pair_df['attrx1'][idx1], sel_pair_df['attry1'][idx1]]) #for stimuli 1 in direction 1
            gt_dir1_stim2 = np.array([sel_pair_df['attrx2'][idx1], sel_pair_df['attry2'][idx1]]) #for stimuli 2 in direction 1
            gt_dir2_stim1 = np.array([sel_pair_df['attrx1'][idx2], sel_pair_df['attry1'][idx2]]) #for stimuli 1 in direction 2
            gt_dir2_stim2 = np.array([sel_pair_df['attrx2'][idx2], sel_pair_df['attry2'][idx2]]) #for stimuli 2 in direction 2

            #theoretical cosine similarity between direction
            gt_dir1 = gt_dir1_stim1 - gt_dir1_stim2
            gt_dir2 = gt_dir2_stim1 - gt_dir2_stim2
            return 1 - scipy.spatial.distance.cosine(gt_dir1,gt_dir2)

        sub_metric_df = []
        for pairtype,sel_pair_df in dir_pairs_dict.items():
            print(f'{subid} - {pairtype}')
            selected_dirs = dirs_arr[np.array(sel_pair_df.index)]
            iter_dir_pairs = list(itertools.combinations(selected_dirs,r=2))
            idx_pairs = list(itertools.combinations(np.array(sel_pair_df.index),r=2))
            

            if np.size(idx_pairs)>0:
                dir_pair_stimgroup = [f"{sel_pair_df['withintrainstim'][idx1]*1}_{1*sel_pair_df['withintrainstim'][idx2]}" for idx1,idx2 in idx_pairs]
                dirp_sg_dict = {
                    '1_1':"Within Training",
                    '1_0':"Training-Test",
                    '0_1':"Training-Test",
                    '0_0':"Within test",
                }
                dir_pair_stimgroup = np.array([dirp_sg_dict[x] for x in dir_pair_stimgroup])

                dir_pair_stimrun = [f"{sel_pair_df['withinrun'][idx1]*1}_{1*sel_pair_df['withinrun'][idx2]}" for idx1,idx2 in idx_pairs]
                dirp_sr_dict = {
                    '1_1':"Within Run",
                    '1_0':"Within-Across",
                    '0_1':"Within-Across",
                    '0_0':"Across Run",
                }
                dir_pair_stimrun = np.array([dirp_sr_dict[x] for x in dir_pair_stimrun])

                theoretical_cos = np.array([compute_theoretical_dir(sel_pair_df,idx1,idx2) for idx1,idx2 in idx_pairs])
                theoretical_PS  = abs(theoretical_cos)
                cos_sim  = np.array([1 - scipy.spatial.distance.cosine(dir1,dir2) for dir1,dir2 in iter_dir_pairs])
                P_scores = abs(cos_sim)

                cos_iter = list(itertools.product(np.unique(theoretical_cos),np.unique(dir_pair_stimgroup),np.unique(dir_pair_stimrun)))
                g_idx_cos = [np.where((theoretical_cos==x) & (dir_pair_stimgroup==g) & (dir_pair_stimrun==r)) for x,g,r in cos_iter]
                cos_iter = [x for x,y in zip(cos_iter,g_idx_cos) if np.size(y)>0 ]
                g_idx_cos = [x for x in g_idx_cos if np.size(x)>0 ]

                PS_iter = list(itertools.product(np.unique(theoretical_PS),np.unique(dir_pair_stimgroup),np.unique(dir_pair_stimrun)))
                g_idx_PS = [np.where((theoretical_PS==x) & (dir_pair_stimgroup==g) & (dir_pair_stimrun==r)) for x,g,r in PS_iter]
                PS_iter = [x for x,y in zip(PS_iter,g_idx_PS) if np.size(y)>0 ]
                g_idx_PS = [x for x in g_idx_PS if np.size(x)>0 ]


                mean_cos_sim = [cos_sim[g_idx].mean() for g_idx in g_idx_cos]
                mean_PS = [P_scores[g_idx].mean() for g_idx in g_idx_PS]

                sub_metric_df.append(
                    pd.DataFrame({
                        "neural":     np.concatenate((mean_cos_sim,mean_PS)),
                        "groundtruth":np.array(cos_iter+PS_iter)[:,0],
                        "withinTraining":np.array(cos_iter+PS_iter)[:,1],
                        "withinRun":np.array(cos_iter+PS_iter)[:,2],
                        "metric":    np.concatenate((["CosineSimilarity"]*len(mean_cos_sim),["ParallelScore"]*len(mean_PS)))
                    }).assign(pairtype = pairtype)
                )

        return pd.concat(sub_metric_df,axis=0).assign(subid=subid)
    
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
    
    def run_ROIPS(self,njobs:int=1):
        with Parallel(n_jobs=njobs) as parallel:
            dfs_list = parallel(
                delayed(self._singleparticipant_ROIPS)(subid) for subid in self.participants)
        PS_df = pd.concat(dfs_list,axis=0) 
        return PS_df