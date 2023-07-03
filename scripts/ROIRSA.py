"""
This script runs RSA analysis in anatomical ROIs

"""

import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import glob
import os
import sys
from joblib import Parallel, delayed, cpu_count

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import ModelRDM, compute_rdm, checkdir, lower_tri, scale_feature
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression


class RSAROI:
    def __init__(self,participants,fmribeh_dir,beta_dir,beta_fname,vsmask_dir,vsmask_fname,anatmasks,nsession) -> None:
        self.participants = participants
        self.fmribeh_dir  = fmribeh_dir
        self.beta_dir     = beta_dir
        self.beta_fname   = beta_fname
        self.vsmask_dir   = vsmask_dir
        self.vsmask_fname = vsmask_fname
        self.anatmasks    = anatmasks
        self.nsession     = nsession
        self.randomseed   = 1
        

    def load_stimlist(self,subid):
        fmribeh_dir = self.fmribeh_dir
        # load stimuli list
        stim_list_fn = glob.glob(os.path.join(fmribeh_dir,subid,'sub*_stimlist.txt'))[0]
        stim_list    =  pd.read_csv(stim_list_fn, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False)
        # get stimuli id
        stim_id = np.array(stim_list['stim_id'])
        # get stimuli image
        stim_image = np.array([x.replace('.png','') for x in stim_list["stim_img"]])
        # get 2d location
        stim_locx = np.array(stim_list['stim_x'])
        stim_locy = np.array(stim_list['stim_y'])
        # get visual features
        stim_color = np.array([x.replace('.png','').split('_')[0] for x in stim_list["stim_img"]])
        stim_shape = np.array([x.replace('.png','').split('_')[1] for x in stim_list["stim_img"]])
        # get training/test stimuli classification
        stim_train = np.array(stim_list['training'])

        stim_dict = {"id":stim_id,
                     "image":stim_image,
                     "locx":stim_locx,
                     "locy":stim_locy,
                     "color":stim_color,
                     "shape":stim_shape,
                     "training":stim_train}
        stimid = stim_dict["image"]
        stimloc = np.vstack([stim_dict["locx"],stim_dict["locy"]]).T
        stimfeature = np.vstack([stim_dict["color"],stim_dict["shape"]]).T
        stimgroup = stim_dict["training"]
        return stimid,stimloc,stimfeature,stimgroup

    def get_modelRDM(self,stimid,stimloc,stimfeature,stimgroup):
        return ModelRDM(stimid = stimid,
                        stimloc = stimloc,
                        stimfeature = stimfeature,
                        stimgroup   = stimgroup,
                        n_session   = self.nsession,
                        randomseed  = self.randomseed)
    
    def rdm_to_df(self,rdm,stimid,group,session,modelname):
        lt,idx = lower_tri(rdm)
        c = np.repeat(np.arange(0, rdm.shape[0]), rdm.shape[0]).reshape(rdm.shape)[idx]
        i = np.tile(np.arange(0, rdm.shape[0]),  rdm.shape[0]).reshape(rdm.shape)[idx]
        df = pd.DataFrame({'stimidA':stimid[c], 'stimidB': stimid[i], 
                           'groupA': group[c],  'groupB': group[i], 
                           'runA': session[c],  'runB': session[i], 
                            modelname: lt}).set_index(['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB'])
        return df
        
    def get_imagedir(self,subid):
        beta_dir = self.beta_dir
        beta_fname = self.beta_fname 
        vsmask_dir = self.vsmask_dir 
        vsmask_fname = self.vsmask_fname

        beta_imgs = [os.path.join(d,'first',subid,f) for d,f in zip(beta_dir,beta_fname)]
        f_masks = [os.path.join(d,'first',subid,f) for d,f in zip(vsmask_dir,vsmask_fname)]

        return beta_imgs,f_masks
    
    def run(self,njobs:int=1):#cpu_count()
        self.njobs = njobs
        with Parallel(n_jobs=self.njobs) as parallel:
            dfs_list = parallel(
                delayed(self.run_singleparticipant)(subid) for subid in self.participants)
        corr_df = pd.concat([x[0] for x in dfs_list],axis=0) 
        rdm_df = pd.concat([x[1] for x in dfs_list],axis=0) 
        return corr_df,rdm_df
    
    
    def run_singleparticipant(self, subid,
               centering:bool=False,group_centering:bool=False):
        
        sys.stderr.write(f"{subid}\r")
        ## compute model rdm
        stimid,stimloc,stimfeature,stimgroup = self.load_stimlist(subid)
        modelrdm  = self.get_modelRDM(stimid,stimloc,stimfeature,stimgroup)

        corr_rdm_names = [x for x in modelrdm.models.keys() if not np.logical_or(x.endswith('session'),x.endswith('within_stimuli'))]
        if self.nsession == 1:
            corr_rdm_names.remove("stimuli")
        corr_rdm_vals = [modelrdm.models[m] for m in corr_rdm_names]

        df_inc_models = [x for x in corr_rdm_names  if not np.logical_or(x.startswith('between_'),x.startswith('within_'))]
        modeldf = modelrdm.rdm_to_df(df_inc_models)

        ## get neural data
        beta_imgs,f_masks = self.get_imagedir(subid)
        mask_imgs = f_masks + self.anatmasks
        APD = ActivityPatternDataLoader(beta_imgs,mask_imgs)
        activitypattern = APD.X
        if centering:
            activitypattern = scale_feature(activitypattern,1,False)
        if group_centering:
            for k in stimgroup:
                activitypattern[np.where(stimgroup == k),:] = scale_feature(activitypattern[np.where(stimgroup == k),:],1,False)

        ## compute neural rdm
        neural_rdm = compute_rdm(activitypattern,'correlation')
        neuralrdmdf = modelrdm.rdm_to_df(modelnames="neural",rdms=neural_rdm)
        neuralrdmdf = neuralrdmdf.join(modeldf).reset_index().assign(subid=subid)
        
        ## compute correlations between neural rdm and model rdms
        PC_s =  PatternCorrelation(neuralrdm = neural_rdm,
                                    modelrdm = corr_rdm_vals,
                                    type="spearman")
        PC_s.fit()
        PC_k =  PatternCorrelation(neuralrdm = neural_rdm,
                                modelrdm = corr_rdm_vals,
                                type="kendall")
        PC_k.fit()
        corr_name_dict = dict(zip(range(len(corr_rdm_vals)),corr_rdm_names))
        corr_df = pd.concat(
            [pd.DataFrame(PC_s.result).T.rename(columns = corr_name_dict).assign(analysis="spearman"),
                pd.DataFrame(PC_k.result).T.rename(columns = corr_name_dict).assign(analysis="kendall")],
                axis=0).assign(subid=subid)
        return corr_df,neuralrdmdf



with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']

ROIRSA_output_path = os.path.join(fmridata_dir,'ROIRSA','navigation_task')
checkdir(ROIRSA_output_path)

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]
anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'
with open(os.path.join(project_path,'scripts','anatomical_masks.json')) as f:    
    anat_roi = list(json.load(f).keys())
laterality = ["left","right","bilateral"]

n_sess = {"fourruns":4,
          "oddeven":2,
          "concatall":1}
for p in preprocess:
    corr_df_list = []
    beta_dir = {
        "fourruns":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')],
        "oddeven":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation_concatodd'),
                    os.path.join(fmridata_dir,p,'LSA_stimuli_navigation_concateven')],
        "concatall":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation_concatall')]
        }
    beta_fname = {
        "fourruns":['stimuli_4r.nii'],
        "oddeven":['stimuli_odd.nii','stimuli_even.nii'],
        "concatall":['stimuli_all.nii']
        }
    vs_dir = {"no_selection":[],
              "reliability_ths0":[os.path.join(fmridata_dir,'unsmoothedLSA','reliability_concat')],
              "perm_rmask":[os.path.join(fmridata_dir,'unsmoothedLSA','reliability_concat')]}
    for ds_name,ds in beta_dir.items():
        for vselect,vdir in vs_dir.items():
            vsmask_dir = beta_dir[ds_name] + vdir
            if vselect =="no_selection":
                vsmask_fname = ['mask.nii']*len(ds)
            elif vselect =="reliability_ths0":
                vsmask_fname = ['mask.nii']*len(ds) + ['reliability_mask.nii']
            elif vselect =="perm_rmask":
                vsmask_fname = ['mask.nii']*len(ds) + ['permuted_reliability_mask.nii']

            rdm_df_list = []
            for roi, lat in itertools.product(anat_roi, laterality):
                print(f"{p} - {ds_name} - {vselect} - {roi} = {lat}")
                anatmasks = [os.path.join(anatmaskdir,f'{roi}_{lat}.nii')]
                testR = RSAROI(subid_list,
                               fmribeh_dir,
                               beta_dir=ds, beta_fname=beta_fname[ds_name],
                               vsmask_dir=vsmask_dir,vsmask_fname=vsmask_fname,
                               anatmasks=anatmasks,
                               nsession=n_sess[ds_name])
                corr_df,rdm_df = testR.run(njobs=cpu_count())
                corr_df = corr_df.assign(roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                corr_df_list.append(corr_df)
                rdm_df = rdm_df.assign(roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                rdm_df_list.append(rdm_df)
            pd.concat(rdm_df_list,axis=0).to_csv(os.path.join(ROIRSA_output_path,f'roirdm_nocentering_{p}_{ds_name}_{vselect}.csv'))
    pd.concat(corr_df_list,axis=0).to_csv(os.path.join(ROIRSA_output_path,f'roicorr_nocentering_{p}.csv'))