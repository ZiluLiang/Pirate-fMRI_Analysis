import itertools
import matlab.engine
import numpy as np
import json
import pandas as pd
import scipy
from copy import deepcopy
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path,'scripts'))
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import compute_rdm,compute_rdm_nomial,compute_rdm_identity, checkdir, scale_feature,lower_tri, upper_tri
from multivariate.modelrdms import ModelRDM
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression,NeuralDirectionCosineSimilarity
from multivariate.rsa_runner import RSARunner

import sklearn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.cross_decomposition import CCA

import torch
from torch import nn


# Directories
with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nonlearners = pirate_defaults['participants']['nonlearnerids']
    nongeneralizers = pirate_defaults['participants']['nongeneralizerids'] + ['sub017']
    
preprocess = ["unsmoothedLSA","smoothed5mmLSA"]            
n_sess = {
          "localizer":1,
          "noconcatall":1,
          "oddeven":2,
          "fourruns":4,
          
          "concatall":1,
          "concateven":1,
          "concatodd":1,
          "concatoddeven":2,          
          
          "fourrunsRESP":4,
          "noconcatallRESP":1,
          "concatallRESP":1,

          "fourrunsBEFORE":4,
          "noconcatallBEFORE":1,
          "concatallBEFORE":1
          }    

LLR_df = pd.read_csv(os.path.join(project_path,'probdf','scanner_test_LLR.csv'))

# ROI
maskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'
with open(os.path.join(project_path,'scripts','anatomical_masks.json')) as f:    
    anat_roi = list(json.load(f).keys())
laterality = ["left","right","bilateral"]
laterality = ["bilateral"]
roi_dict = dict(zip(
    ["frontal","ofc","hippocampus","parahippocampus","parietal","occipital"],
    ["frontal","ofc","hippocampus","parahippocampus","parietal","occipital"]
    ))
ROIRSA_output_path = os.path.join(fmridata_dir,'ROIRSA','anatomic_ROIs')
checkdir(ROIRSA_output_path)

# Extract data
neuralrdm_rops,behav_dfs = dict(), dict()
for p in preprocess[:1]:
    beta_dir = {
        "oddeven":[os.path.join(fmridata_dir,p,'LSA_stimuli_navigation')]*2,
        #"localizer":[os.path.join(fmridata_dir,p,'LSA_stimuli_localizer')]
        }
    beta_fname = {
        "oddeven":['stimuli_odd.nii',
                   'stimuli_even.nii'],
        "localizer":['stimuli_1r.nii'],
        }
    vs_dir = {
              "no_selection":[],
              }
    for ds_name,ds in list(beta_dir.items()):
        neuralrdm_rops[ds_name] = dict()
        behav_dfs[ds_name] = dict()
        for vselect,vdir in vs_dir.items():
            mds_df_list = []
            rdm_df_list = []
            PS_df_list = []

            vsmask_dir = ds + vdir
            if vselect == "no_selection":
                vsmask_fname = ['mask.nii']*len(ds)
            elif vselect == "perm_rmask":
                vsmask_fname = ['mask.nii']*len(ds) + ['permuted_reliability_mask.nii']
            elif vselect == "reliability_ths0":
                vsmask_fname = ['mask.nii']*len(ds) + ['reliability_mask.nii']
            taskname = "navigation" if ds_name != "localizer" else ds_name
            
            with Parallel(n_jobs=4) as parallel:
                for (roi,roi_fn), lat in itertools.product(roi_dict.items(), laterality):
                    print(f"{p} - {ds_name} - {vselect} - {roi} = {lat}")
                    anatmasks = [os.path.join(maskdir,f'{roi_fn}_{lat}.nii')]
                    #anatmasks = [os.path.join(maskdir,f'{roi_fn}.nii')]
                    RSA = RSARunner(participants=subid_list,
                                    fmribeh_dir=fmribeh_dir,
                                    beta_dir=ds, beta_fname=beta_fname[ds_name],
                                    vsmask_dir=vsmask_dir, vsmask_fname=vsmask_fname,
                                    pmask_dir=ds, pmask_fname=beta_fname[ds_name],
                                    anatmasks=anatmasks,
                                    nsession=n_sess[ds_name],
                                    taskname=taskname)
                    neuralrdm_rops[ds_name][roi] = parallel(delayed(RSA.get_neuralRDM)(subid) for subid in subid_list)
                sub_mrdms = parallel(delayed(RSA.get_modelRDM)(subid) for subid in subid_list)
                behav_dfs[ds_name] = [m.stimdf.assign(subid=subid,stim_sessionoddeven=(m.stimdf["stim_session"]+1)%2) for m,subid in zip(sub_mrdms,subid_list)]


# MLP DECODER

class MLP(nn.Module):
    def __init__(self,n_input,n_hidden=20,n_output=2):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(n_input,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_output)
        )
    def forward(self,x):
        return self.layers(x)

def training(model,training_x,training_y,evaluation_x,evaluation_y,optimizer,loss_function,n_epoch=1000):
    training_loss = []
    loss_training,loss_evaluation = [], []
    for epoch in range(n_epoch):        
        predict_training = model.forward(training_x)
        loss = loss_function(predict_training,training_y)
        training_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        with torch.no_grad():
            predict_training, predict_evaluation = model.forward(training_x), model.forward(evaluation_x)
            loss_training.append(loss_function(predict_training,training_y).item())
            loss_evaluation.append(loss_function(predict_evaluation,evaluation_y).item())
        
    return loss_training,loss_evaluation,predict_training, predict_evaluation

def MLP_decoding_cv(X,stimdf,readout_colnames,splitgroup_colnames,n_hidden = 5,n_epoch=100):
    
    X_preprocessed=scale_feature(X,2)
    raw_readoutfeatures = np.array(stimdf[readout_colnames])
    readoutfeatures = raw_readoutfeatures 

    loss_function = nn.MSELoss()

    logo=LeaveOneGroupOut()
    groups=np.array(stimdf[splitgroup_colnames]) 

    loss_fit_splits,loss_eval_splits = [],[]
    score_fit_splits,score_eval_splits = [],[]
    pred_fit_splits,pred_eval_splits = [],[]
    true_fit_splits,true_eval_splits = [],[]
    for j,(fit_idx,eval_idx) in list(enumerate(logo.split(X=X_preprocessed,groups=groups)))[:]:
        model=MLP(n_input=X_preprocessed.shape[1],n_hidden=n_hidden,n_output=readoutfeatures.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        
        fit_x,eval_x = torch.tensor(X_preprocessed[fit_idx,:]).to(torch.float32), torch.tensor(X_preprocessed[eval_idx,:]).to(torch.float32)
        fit_y,eval_y = torch.tensor(readoutfeatures[fit_idx,:]).to(torch.float32), torch.tensor(readoutfeatures[eval_idx,:]).to(torch.float32)

        loss_fit,loss_eval,fit_pred, eval_pred = training(model,fit_x,fit_y,eval_x,eval_y,optimizer,loss_function,n_epoch)

        loss_fit_splits.append(loss_fit)
        loss_eval_splits.append(loss_eval)
        
        score_fit_splits.append(sklearn.metrics.r2_score(fit_y,fit_pred))
        score_eval_splits.append(sklearn.metrics.r2_score(eval_y,eval_pred))
        
        pred_fit_splits.append(fit_pred)
        pred_eval_splits.append(eval_pred)
        true_fit_splits.append(fit_y)
        true_eval_splits.append(eval_y)

    predictions = np.concatenate([np.concatenate(pred_fit_splits,axis=0),np.concatenate(pred_eval_splits,axis=0)],axis=0)
    observations= np.concatenate([np.concatenate(true_fit_splits,axis=0),np.concatenate(true_eval_splits,axis=0)],axis=0)
    stage = np.array(["fit"]*np.concatenate(pred_fit_splits,axis=0).shape[0] + ["eval"]*np.concatenate(pred_eval_splits,axis=0).shape[0])

    loss = {"fit":np.array(loss_fit_splits),
              "eval":np.array(loss_eval_splits)}
    scores = {"fit":np.mean(score_fit_splits),
              "eval":np.mean(score_eval_splits)}
    behavior = {"predictions": predictions,
                 "observations":observations,
                 "stage":stage}
    return loss,scores,behavior
# shuffle method
def shuffle_activitypatternmatrix(X,seed=None):
    rng = np.random.default_rng(seed)
    Xshuffle = np.array([X[rng.permutation(X.shape[0]),j] for j in range(X.shape[1])]).T
    return Xshuffle

############################### loop over stimuli category and roi to generate null distribution of decoding performance
n_perm = 1000
ds_name="oddeven"

filter_teststim = [np.array(bdf['stim_group'])!=1 for bdf in behav_dfs[ds_name]]
filter_allstim = [np.array(bdf['stim_group'])!=2 for bdf in behav_dfs[ds_name]]

decode_type = {
    ######### cross condition decoding  ################
    "decode-stimy_cross-stimx":{"readout":["stim_y"], "splitby":"stim_x"},
    "decode-stimx_cross-stimy":{"readout":["stim_x"], "splitby":"stim_y"},
    "decode-respy_cross-stimx":{"readout":["resp_y"], "splitby":"stim_x"},
    "decode-respx_cross-stimy":{"readout":["resp_x"], "splitby":"stim_y"},

    ######### cross group decoding  ################
   "decode-stimxy_cross-group":{"readout":["stim_x","stim_y"],"splitby":"stim_group"},
   "decode-respxy_cross-group":{"readout":["resp_x","resp_y"],"splitby":"stim_group"},
    
    ######### cross run decoding  ################
   "decode-stimxy_cross-run":{"readout":["stim_x","stim_y"],"splitby":"stim_session"},
   "decode-respxy_cross-run":{"readout":["resp_x","resp_y"],"splitby":"stim_session"},

}

with Parallel(n_jobs=10) as parallel:
    for dcode_name,dcode_cfg in decode_type.items():
        print(f"{dcode_name}")
        readout_colnames = dcode_cfg["readout"]
        splitgroup_colnames=dcode_cfg["splitby"]

        if "cross-group" not in dcode_name:
            filters = {
                "all stimuli": filter_allstim,
                "test stimuli": filter_teststim
            }
        else:
            filters = {
                "all stimuli": filter_allstim,
            }

        perm_sub_fit_err,perm_sub_eval_err = {},{}
        for flist_name, flist in filters.items():
            perm_sub_fit_err[flist_name],perm_sub_eval_err[flist_name] = {},{}
            for roi in roi_dict.keys():
                print(f"{flist_name}-{roi}")
                perm_sub_fit_err[flist_name][roi], perm_sub_eval_err[flist_name][roi] = [], []
                for k in range(n_perm):
                    decoder_outputs_shuffle = parallel(delayed(MLP_decoding_cv)(
                                                        shuffle_activitypatternmatrix(op[0][f,:]),
                                                        bdf[f].reset_index(),
                                                        readout_colnames,splitgroup_colnames,n_hidden=10,n_epoch=100) for subid,bdf,op,f in zip(subid_list,behav_dfs[ds_name],neuralrdm_rops[ds_name][roi],flist))
                    perm_sub_fit_err[flist_name][roi].append([o[1]["fit"]for o in decoder_outputs_shuffle])
                    perm_sub_eval_err[flist_name][roi].append([o[1]["eval"]for o in decoder_outputs_shuffle])
        output_dir = os.path.join(ROIRSA_output_path,"decoding","navigation")
        checkdir(output_dir)
        fn = os.path.join(output_dir,f"{dcode_name}_permutateXcols_{n_perm}perms.pkl")
        dump({"fit": perm_sub_fit_err,"eval":perm_sub_eval_err},filename=fn)