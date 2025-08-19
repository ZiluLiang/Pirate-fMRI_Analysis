"""
This file computes the retrieval pattern of test stimuli from training stimuli/location with cross-validation.
"""
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import json
from copy import deepcopy
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri, compute_rdm, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature,concat_data,kabsch_algorithm
from zpyhelper.MVPA.estimators import MetaEstimator

from typing import Union
import sys
import os

from scipy.linalg import orthogonal_procrustes
project_path = r"D:\OneDrive - Nexus365\pirate_ongoing"
sys.path.append(project_path)

import warnings
warnings.simplefilter('ignore', category=FutureWarning)


# Load data
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

ROIRSAdir = os.path.join(project_path,'AALandHCPMMP1_anatrepfun')
##################################################### METHODS #####################################################

def average_and_reorder(activitypattern,stimdf):
        
        # filter out data from treasure hunt task
        navi_filter = np.vstack(
            [stimdf.taskname.to_numpy() == "navigation",
             [not np.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]]
        ).all(axis=0)

        navi_X = activitypattern[navi_filter,:]
        assert navi_X.shape[0] == 24*4 # because we are ignoring the central sitmuli here, so we have (25-1)*4 stimuli in total
        navi_df = stimdf[navi_filter].copy().reset_index(drop=True)

        #then we process the treasure hunt task data here: we get the average of odd and even splits
        navi_Xsess = split_data(navi_X,navi_df.stim_session.to_numpy())
        oddX,evenX = np.mean([navi_Xsess[0],navi_Xsess[2]],axis=0), np.mean([navi_Xsess[1],navi_Xsess[3]],axis=0)
        # then we z-score for each split
        oddX,evenX = scale_feature(oddX,2), scale_feature(evenX,2)
        # and we put them back into a matrix
        navi_X = concat_data([oddX,
                              evenX
                             ])
        navi_df = navi_df[navi_df.stim_session<2].copy().reset_index(drop=True)

        ## split into training and test for reordering: this is to make sure the final weight matrices are ordered in the way we want
        training_filter = navi_df.stim_group==1
        test_filter = navi_df.stim_group==0


        trdf = navi_df[training_filter].copy().reset_index(drop=True)
        trX  = navi_X[training_filter]
        trdfneworder = trdf.sort_values(by=['stim_session','training_axset','training_axlocTL'])
        new_order = trdfneworder.index
        training_df = trdfneworder.reset_index(drop=True).assign(stim_task=0)
        training_X = trX[new_order,:]
        ave_trX = np.mean(split_data(training_X,training_df.stim_session.to_numpy()),axis=0)
        
        teX = navi_X[test_filter]
        tedf = navi_df[test_filter].copy().reset_index(drop=True)
        teneworder = tedf.sort_values(by=['stim_session','stim_x','stim_y'])
        teneworderidx = teneworder.index
        test_X = teX[teneworderidx,:]
        test_df = teneworder.reset_index(drop=True).assign(stim_task=0)

        # filter out data from localizer task
        lzer_filter = np.vstack(
            [stimdf.taskname.to_numpy() == "localizer",
            [not np.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]]
        ).all(axis=0)       
        lzerX  = activitypattern[lzer_filter,:]
        # then we z-score for each split (only 1 split for localizer)
        lzerX = scale_feature(lzerX,2)
        lzdfneworder = stimdf[lzer_filter].copy().reset_index(drop=True).sort_values(by=['stim_session','training_axset','training_axlocTL'])
        new_order = lzdfneworder.index
        lzer_df = lzdfneworder.reset_index(drop=True).assign(stim_task=1)
        lzer_X = lzerX[new_order,:]
        return [training_X,test_X,lzer_X],  [training_df,test_df,lzer_df]

def navidata_OEsplitter(stimdf):
    assert np.logical_and(stimdf.stim_task.nunique()==1,stimdf.stim_task.unique()[0]==0), f"must be navigation task(0), got {stimdf.stim_task.unique()} instead"
    odd_f  = stimdf.stim_session.to_numpy() == 0
    even_f = stimdf.stim_session.to_numpy() == 1
    return odd_f,even_f


class CompositionalRetrieval_CV_Reduced(MetaEstimator):
    def __init__(self,activitypatterns:list,
                 stim_dfs:list,
                 run_reg=None) -> None:
        
        #get the preprocessed and reordered data
        [navitrain_X,navitest_X,lzer_X], [navitrain_df,navitest_df,lzer_df] = activitypatterns, stim_dfs

        if lzer_df.stim_session.nunique() == 1:
            self.splitlzer = False
        else:
            self.splitlzer = True
        
        self.Xs = {
            "trainstim": navitrain_X,
            "teststim":  navitest_X,
            "locations": lzer_X,
        }

        self.dfs = {
            "trainstim": navitrain_df,
            "teststim":  navitest_df,
            "locations": lzer_df,
        }

        self.regression_configs = {
            "train2test": ["trainstim","teststim"], # in the order of independent var, dependent var
            "loc2test": ["locations","teststim"],
            "loc2train": ["locations","trainstim"]
        }
        
        if run_reg is None:
            run_reg = list(self.regression_configs.keys())
        else:
            run_reg = [x for x in run_reg if x in self.regression_configs.keys()]
        assert len(run_reg)>0, "No valid regression configurations specified"
        self.run_reg = run_reg

    def data_OEsplitter(self,stimdf):
        assert stimdf.stim_task.nunique()==1,f"Can only split data for one task at a time, got {stimdf.stim_task.unique()} tasks instead"
        uniqueS = stimdf.stim_session.unique()
        assert uniqueS.size==2,f"expecting 2unique sessions, got {uniqueS.size} instead"
        odd_f  = stimdf.stim_session.to_numpy() == uniqueS[0]
        even_f = stimdf.stim_session.to_numpy() == uniqueS[1]
        return odd_f,even_f


    def fit(self):
        
        result = []
        result_names = []
        
        regcoefmats = {}
        cv_split_reg_coefs = {}
        pred_configurals = {}
        for regname in self.run_reg:
            rcfg = self.regression_configs[regname]
            component_X,  configural_X = self.Xs[rcfg[0]], self.Xs[rcfg[1]]
            component_df, configural_df = self.dfs[rcfg[0]], self.dfs[rcfg[1]]
            if regname=="train2test":
                # if train2test do for each odd-even split, and see if the results are consistent
                conf_odd_fil, conf_even_fil = self.data_OEsplitter(configural_df)
                comp_odd_fil, comp_even_fil = self.data_OEsplitter(component_df)
                
            else:
                # if lzer2navi, only split for navigation data and see if results are consistent
                conf_odd_fil, conf_even_fil = self.data_OEsplitter(configural_df)
                if self.splitlzer: 
                    comp_odd_fil, comp_even_fil = self.data_OEsplitter(component_df)
                else:
                    comp_odd_fil, comp_even_fil = component_df.stim_task.values == 1,component_df.stim_task.values == 1
            

            splitter = {"O2E": [[conf_odd_fil,comp_odd_fil],   [conf_even_fil,comp_even_fil]],
                        "E2O": [[conf_even_fil,comp_even_fil], [conf_odd_fil,comp_odd_fil]]
                        }

            reg_res = {}
            cv_split_reg_coefs[regname] = []
            
            cv_split_preds = []
            
            for cvsplit,[[fitf_tar,fitf_X],[evalf_tar,evalf_X]] in splitter.items():
                # here, targets are the activity patterns of different configural stimuli
                # features are the activity patterns of different component stimuli
                # so voxels are samples, stimuli are features/targets:
                # fit_tar,eval_tar are of shape  nvoxel * n_configural_stimuli
                # fit_X,eval_X are of shape nvoxel * n_component_stimuli
                fit_tar,  fit_X  = configural_X[fitf_tar,:].T, (component_X[fitf_X,:]).T 
                eval_tar, eval_X = configural_X[evalf_tar,:].T, (component_X[evalf_X,:]).T

                # make sure there is no nan bc LinearRegression does not like it
                assert np.isnan(configural_X).sum() == 0, f"{regname} configural_X has nan values {configural_X}"
                assert np.isnan(component_X).sum() == 0, f"{regname} component_X has nan values {component_X}"
                
                # fit the weights on the fit set
                reg_estimator = LinearRegression(fit_intercept=True,positive=False).fit(fit_X,fit_tar)
                # predict on the fit and eval set
                fit_pred, eval_pred = reg_estimator.predict(fit_X), reg_estimator.predict(eval_X)
                # get the weights
                coef_mat = reg_estimator.coef_
                
                cv_split_preds.append(eval_pred)
                cv_split_reg_coefs[regname].append(coef_mat)
                reg_res[f"fit_r2_{cvsplit}"]  = r2_score(fit_tar,fit_pred)
                reg_res[f"eval_r2_{cvsplit}"] = r2_score(eval_tar,eval_pred)                
                reg_res[f"fit_corr_{cvsplit}"] = np.mean([pearsonr(tarpat,predpat).statistic for tarpat,predpat in zip(fit_tar.T,fit_pred.T)]) 
                reg_res[f"eval_corr_{cvsplit}"] = np.mean([pearsonr(tarpat,predpat).statistic for tarpat,predpat in zip(eval_tar.T,eval_pred.T)]) 
                
            for metric in ["r2","corr"]:
                reg_res[f"fit_{metric}"]  = np.mean([reg_res[f"fit_{metric}_{cvsplit}"] for cvsplit in splitter.keys()])
                reg_res[f"eval_{metric}"] = np.mean([reg_res[f"eval_{metric}_{cvsplit}"] for cvsplit in splitter.keys()])            

            reg_coefs = np.mean(cv_split_reg_coefs[regname],axis=0)
            for cvsplit in splitter.keys():
                reg_res.pop(f"fit_r2_{cvsplit}")
                reg_res.pop(f"eval_r2_{cvsplit}")
            
            
            component_df  = component_df[comp_odd_fil].copy()
            configural_df = configural_df[conf_odd_fil].copy()
            
            if "loc2train" in regname:
                percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                      component_df[["stim_x","stim_y"]].to_numpy(),
                                      lambda u,v: 1*np.array_equal(u,v))
                dist_mod_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                       component_df[["stim_x","stim_y"]].to_numpy(),
                                        lambda u,v: np.sqrt(2)-np.linalg.norm(u-v))
            else:
                percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                        component_df[["stim_x","stim_y"]].to_numpy(),
                                        lambda u,v: sum(u==v))
                dist_mod_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                        component_df[["stim_x","stim_y"]].to_numpy(),
                                        lambda u,v: np.sum((2-np.abs(u-v))*(1*(v!=0))))
                
            #compo_locrep  = component_df[["stim_x","stim_y"]].to_numpy()*2
            #config_locrep = configural_df[["stim_x","stim_y"]].to_numpy()*2
            #loc_reg = LinearRegression(fit_intercept=False,positive=False).fit(compo_locrep.T, config_locrep.T)
            #dist_mod_reg_coefs = loc_reg.coef_
            
            pdcoef = LinearRegression(fit_intercept=True).fit(
                        X=np.array([scale_feature(dist_mod_reg_coefs.flatten(),2),scale_feature(percept_reg_coefs.flatten(),2)]).T,
                        y=scale_feature(reg_coefs.flatten(),2)
                    ).coef_
            #pdcoef = np.mean(
            #    [LinearRegression(fit_intercept=True).fit(X=np.vstack([scale_feature(dm,2),scale_feature(pm,2)]).T,y=scale_feature(rm,2)).coef_ for dm,pm,rm in zip(dist_mod_reg_coefs,percept_reg_coefs,reg_coefs)],
            #    axis=0
            #)

            regcoefmats[regname] = [reg_coefs,percept_reg_coefs,dist_mod_reg_coefs]
            pred_configurals[regname] = np.mean(cv_split_preds,axis=0)
            

            n_rel = 8 if "loc2train" in regname else 16*2
            compo_stims = reg_coefs[percept_reg_coefs==1]
            compo_stims_x = reg_coefs[:,:4][percept_reg_coefs[:,:4]==1]
            compo_stims_y = reg_coefs[:,4:][percept_reg_coefs[:,4:]==1]
            #print(compo_stims.shape)
            noncompo_stims = reg_coefs[percept_reg_coefs!=1]
            noncompo_stims_x = reg_coefs[:,:4][percept_reg_coefs[:,:4]==0]
            noncompo_stims_y = reg_coefs[:,4:][percept_reg_coefs[:,4:]==0]

            assert compo_stims.size == n_rel, f"Expecting {n_rel} got {compo_stims.size} relevant stimuli, {percept_reg_coefs}"
            assert noncompo_stims.size == reg_coefs.size - n_rel, f"Expecting {reg_coefs.size - n_rel} got {noncompo_stims.size} non relevant stimuli"
            assert compo_stims_x.size == int(n_rel/2),  f"Expecting {int(n_rel/2)} got {compo_stims_x.size} relevant stimuli on x"
            assert noncompo_stims_x.size == int((reg_coefs.size - n_rel)/2)

            result = result + list(reg_res.values()) + \
                        [pdcoef[0], pdcoef[1], 
                         compo_stims_x.mean(), compo_stims_y.mean(), compo_stims.mean(), 
                         noncompo_stims_x.mean(), noncompo_stims_y.mean(), noncompo_stims.mean(),
                         compo_stims.mean()-noncompo_stims.mean(), compo_stims.sum()-noncompo_stims.sum()]
            prim_names = ["reg_distmod","reg_percept",
                          "compoweight_x","compoweight_y","compoweight", 
                          "noncompoweight_x", "noncompoweight_y","noncompoweight",
                          "meanweightdiff", "sumweightdiff"]
            result_names = result_names + [f"{regname}-{x}" for x in reg_res.keys()] + [f"{regname}-{x}" for x in prim_names]

            
        self.heldoutpred_configurals = pred_configurals
        self.cv_split_reg_coefs = cv_split_reg_coefs    
        self.result = result
        self.resultnames = result_names
        self.regcoefmats = regcoefmats
        return self
    
    def visualize(self):
        fig = self.estimator.visualize()
        return fig
    
    def __str__(self) -> str:
        return f"CompositionalRetrieval"
    
    def get_details(self)->str:
        details = {
            "name": self.__str__(),
            "resultnames": list(self.resultnames)
        }
        return details

##################################################### ANALYSIS STARTS HERE #####################################################
organize_and_get_data = False
n_perm = 100
run_sc = True
run_ana = True
rois = ["PPC_bilateral","V1_bilateral","vmPFC_bilateral","HPC_bilateral"][:1]#["testgtlocParietalSup_bilateral","vmPFC_bilateral","HPC_bilateral","V1_bilateral"]
restuldir = os.path.join(ROIRSAdir,"retrievalpatternCV_rotated_new")
checkdir(restuldir)


if organize_and_get_data:
    roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
    #rois = list(roi_data.keys())
    
    # first we get the averaged and reordered data
    allsubXs, allsubdfs = {},{}
    for roi in rois:
        allsubXs[roi], allsubdfs[roi] = [],[]
        for subdata,subid in zip(roi_data[roi],subid_list):
            print(f"{roi} - {subid}",end="\r",flush=True)
            # reorder and get the data for each pariticpnat
            Xs, dfs = average_and_reorder(deepcopy(subdata["preprocX"]),subdata["stimdf"].copy())
            allsubXs[roi].append(Xs)
            allsubdfs[roi].append(dfs)


    # Get the data for rotated lzer and rotated shuffled lzer
    import time
    with Parallel(n_jobs=20) as parallel:
        print("Rotating localizer data")
        for roi in rois:
            # run and save for each ROI
            rotated_lzerX, shufflelzerX, rotated_shufflelzerX, rotated_lzerdfs = [], [], [], []
            subXs, subdfs = allsubXs[roi],allsubdfs[roi]#[], []
            for Xs,dfs,subid in zip(allsubXs[roi],allsubdfs[roi],subid_list):
                
                trX, _, lzer_X = Xs
                trdf, _, lzer_df = dfs
                
                # get the shuffle of lzer: we permuted the rows, i.e., stimuli labels were randomly reassigned to activity patterns                
                starttime= time.time()
                lzer_shuffled_Xs = [np.random.default_rng(p).permutation(lzer_X) for p in range(n_perm)]
                print(f"Shuffling {n_perm} localizer data in {roi} - {subid} took {time.time()-starttime:.2f}s",end="\r",flush=True)
                

                # split training stim data into odd and even
                trfodd, trfeven = navidata_OEsplitter(trdf)

                # align lzer and all the shuffle lzer to odd and even tr using the central stimuli
                starttime= time.time()

                Rlzerodd, _,  _ = kabsch_algorithm(lzer_X, trX[trfodd])
                Rlzereven, _, _ = kabsch_algorithm(lzer_X, trX[trfeven])
                lzeraligned_stacked  = np.vstack(
                                           [Rlzerodd,
                                            Rlzereven
                                           ]
                                           )
                odd_res = parallel(delayed(kabsch_algorithm)(P,trX[trfodd]) for P in lzer_shuffled_Xs)
                even_res = parallel(delayed(kabsch_algorithm)(P,trX[trfeven]) for P in lzer_shuffled_Xs)
                slzeraligned_stacked = [np.vstack([lso[0],lse[0]]) for lso,lse in zip(odd_res,even_res)]

                print(f"Rotating {n_perm} (ori+permutated) localizer data in {roi} - {subid} took {time.time()-starttime:.2f}s",end="\r",flush=True)

                rotated_lzerX.append(lzeraligned_stacked)
                shufflelzerX.append(lzer_shuffled_Xs)
                rotated_shufflelzerX.append(slzeraligned_stacked)
                rotated_lzerdfs.append(
                    # the original lzer_df only has one session,
                    # because we align lzer pattern twice, once to odd and once to even, 
                    # the rotated lzer_df now has two sessions, so we need to concat them
                    pd.concat([lzer_df.assign(stim_session=0), lzer_df.assign(stim_session=1)]).reset_index(drop=True)
                )

            checkdir(os.path.join(restuldir,roi))
            print(f"\nSaving rotated data for {roi}")
            dump({"unrotated_Xs":subXs,
                "unrotated_dfs":subdfs,
                "rotated_lzerX":rotated_lzerX,
                "shuffled_lzerX":shufflelzerX,
                "rotated_shufflelzerX":rotated_shufflelzerX,
                "rotated_lzerdfs":rotated_lzerdfs
                },
                os.path.join(restuldir,roi,f"rotated_datasets.pkl")
                )


if run_sc:
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score
    compare_rdm = lambda X0,X1: pearsonr(lower_tri(compute_rdm(X0,"correlation"))[0], lower_tri(compute_rdm(X1,"correlation"))[0]).statistic
    rdm_sc = {"odd":{}, "even":{},"shuffledodd":{}, "shuffledeven":{}}
    rdm_corrOE = {"train":{},"test":{},"all":{}}
    r2OE = {"train":{},"test":{},"all":{}}
    roi_shortnames = {
        "PPC_bilateral":"PPC",
        "V1_bilateral":"V1"
    }
    roi_shortnames = dict(zip(rois,
                              [x.replace("_bilateral","") for x in rois]
                              ))
    for roi in rois:
        rotated_datasets = load(os.path.join(restuldir,roi,"rotated_datasets.pkl"))
        subXs, subdfs = rotated_datasets["unrotated_Xs"], rotated_datasets["unrotated_dfs"]
        rotated_lzerX, rotated_shufflelzerX, rotated_lzerdfs = rotated_datasets["rotated_lzerX"], rotated_datasets["rotated_shufflelzerX"], rotated_datasets["rotated_lzerdfs"]
        # correlation between odd and even
        rdm_corrOE["all"][roi] = [compare_rdm(np.vstack([Xs[1][dfs[1].stim_session.values==0], Xs[0][dfs[0].stim_session.values==0]]),
                                              np.vstack([Xs[1][dfs[1].stim_session.values==1], Xs[0][dfs[0].stim_session.values==1]])) for Xs,dfs in zip(subXs, subdfs)]
        r2OE["all"][roi] = [r2_score(np.vstack([Xs[1][dfs[1].stim_session.values==0], Xs[0][dfs[0].stim_session.values==0]]),
                                      np.vstack([Xs[1][dfs[1].stim_session.values==1], Xs[0][dfs[0].stim_session.values==1]])) for Xs,dfs in zip(subXs, subdfs)]
        rdm_corrOE["test"][roi] = [compare_rdm(Xs[1][dfs[1].stim_session.values==0],
                                               Xs[1][dfs[1].stim_session.values==1]) for Xs,dfs in zip(subXs, subdfs)]
        r2OE["test"][roi] = [r2_score(Xs[1][dfs[1].stim_session.values==0],
                                      Xs[1][dfs[1].stim_session.values==1]) for Xs,dfs in zip(subXs, subdfs)]

        # run sanity checks to see if rotation changed the representation structure
        rdm_sc["odd"][roi]  = [compare_rdm(unrX[2], rX[rdf.stim_session.values==0]) for unrX,rX,rdf in zip(subXs,rotated_lzerX, rotated_lzerdfs)]
        rdm_sc["even"][roi] = [compare_rdm(unrX[2], rX[rdf.stim_session.values==1]) for unrX,rX,rdf in zip(subXs,rotated_lzerX, rotated_lzerdfs)]

        rdm_sc["shuffledodd"][roi] = [[compare_rdm(srX[rdf.stim_session.values==0], rX[rdf.stim_session.values==0]) for srX in srXs] for srXs,rX,rdf in zip(rotated_shufflelzerX,rotated_lzerX, rotated_lzerdfs)]
        rdm_sc["shuffledeven"][roi] = [[compare_rdm(srX[rdf.stim_session.values==1], rX[rdf.stim_session.values==1]) for srX in srXs] for srXs,rX,rdf in zip(rotated_shufflelzerX,rotated_lzerX, rotated_lzerdfs)]

    from scripts.Exp1_fmri.plotting_utils import gen_pval_annot
    choose_color_from = sns.color_palette("colorblind",10).as_hex()
    participantgrouphex = dict(zip(
                    ["Generalizer","nonGeneralizer"],
                    [choose_color_from[2],choose_color_from[4]]  
                ))
    for roi in rois:
        fig = plt.figure(figsize=(10,5))
        plt.subplot(1,2,2)
        print(roi)
        corrdata = [x for x,subid in zip(rdm_corrOE["all"][roi],subid_list) if subid in generalizers]
        r2data = [x for x,subid in zip(r2OE["all"][roi],subid_list) if subid in generalizers]
        corr_perm_res = scipy.stats.permutation_test((np.arctanh(corrdata),),
                                                     statistic=np.mean,permutation_type="samples",
                                                     random_state=0,n_resamples=10000)
        print(corr_perm_res)

        plt.hist(corrdata,color=participantgrouphex["Generalizer"],alpha=0.4)
        plt.xlabel("correlation", fontsize=12,fontweight="bold")
        plt.ylabel("count", fontsize=12,fontweight="bold")
        plt.axvline(np.mean(corrdata),color=participantgrouphex["Generalizer"],linestyle="--",label="mean")
        plt.text(0.5,0.9,f"mean rho={np.mean(corrdata):.3f}{gen_pval_annot(corr_perm_res.pvalue,False)}",transform=plt.gca().transAxes,fontsize=12,fontweight="bold")
        plt.title("Correlation \nbetwen RDM of odd and even data", fontsize=14,fontweight="bold")
        plt.subplot(1,2,1)
        plt.hist(r2data,color=participantgrouphex["Generalizer"],alpha=0.4)
        plt.xlabel("R-square", fontsize=12,fontweight="bold")
        plt.ylabel("count", fontsize=12,fontweight="bold")
        plt.title("Coefficient of determination \nbetwen odd and even data", fontsize=14,fontweight="bold")
        fig.suptitle(f"{roi_shortnames[roi]}", fontsize=16,fontweight="bold")
        fig.tight_layout()

    for roi in rois:
        
        scodddata = [x for x,subid in zip(rdm_sc["odd"][roi],subid_list) if subid in generalizers]
        scevendata = [x for x,subid in zip(rdm_sc["even"][roi],subid_list) if subid in generalizers]
        xlims = (np.min(np.concatenate([scodddata,scevendata]))*0.999,
                 np.max(np.concatenate([scodddata,scevendata])))
        ticks = np.round(np.linspace(xlims[0],xlims[1],num=5,endpoint=True),4)
        fig = plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.hist(scodddata,color=participantgrouphex["Generalizer"],alpha=0.4)
        plt.xlim(xlims)
        plt.xticks(ticks)
        plt.xlabel("correlation", fontsize=12,fontweight="bold")
        plt.ylabel("count", fontsize=12,fontweight="bold")
        plt.title("Aligned to training stim. from odd", fontsize=14,fontweight="bold")
        
        plt.subplot(1,2,2)
        plt.hist(scevendata,color=participantgrouphex["Generalizer"],alpha=0.4)
        plt.xlim(xlims)
        plt.xticks(ticks)
        plt.xlabel("correlation", fontsize=12,fontweight="bold")
        plt.ylabel("count", fontsize=12,fontweight="bold")
        plt.title("Aligned to training stim. from even", fontsize=14,fontweight="bold")
        fig.suptitle(f"Before and After Rotation Alignment Localizer Task RDM Correlation\n{roi_shortnames[roi]}", fontweight="bold", fontsize=16) 
        fig.tight_layout()
        #plt.hist([np.mean(x) for x in rdm_sc["shuffledodd"][roi]],color=participantgrouphex["Generalizer"],alpha=0.4)
        #plt.hist([np.mean(x) for x in rdm_sc["shuffledeven"][roi]],color=participantgrouphex["Generalizer"],alpha=0.4)

# run the analysis
if run_ana:
    print("\nRunning analysis\n")
    def RPCV_run_group(subXs,subdfs,subid_list,run_regs):
        results = []
        reg_coefs = dict(zip(run_regs,[[] for _ in run_regs]))
        percept_reg_coefs, dmod_reg_coefs = {}, {}

        for Xs,dfs,subid in zip(subXs,subdfs,subid_list):
            CRestimator = CompositionalRetrieval_CV_Reduced(
                                activitypatterns=Xs,
                                stim_dfs=dfs,
                                run_reg=run_regs
                                )
            CRestimator.fit()

            subresdf = pd.DataFrame({"value":CRestimator.result,
                                    "metric":CRestimator.resultnames}
                                    ).assign(roi=roi,subid=subid)
            results.append(subresdf)
            for ana in run_regs:
                reg_coefs[ana].append(CRestimator.regcoefmats[ana][0])
        # save the model retrieval pattern from last participant for double-checking
        for ana in run_regs:
            percept_reg_coefs[ana] = CRestimator.regcoefmats[ana][1]
            dmod_reg_coefs[ana] = CRestimator.regcoefmats[ana][2]
        resultdf = pd.concat(results,axis=0).reset_index(drop=True)
        return reg_coefs,resultdf

    def runrsaforshuffledlzer(p,Xs,dfs,subid_list,run_regs,outputdir):
        print(f"\nRunning Rotated Shuffle {p}",end="\r",flush=True)
        _, resultdf   = RPCV_run_group(Xs,dfs,subid_list,run_regs) 
        curr_perm_res = resultdf.assign(roi=roi,dataset="rotatedshuffle",perm=p)
        curr_perm_res.to_csv(os.path.join(outputdir,f"perm{p}_results.csv"),index=False)

    with Parallel(n_jobs=10) as parallel:
        for roi in rois:
            rotated_datasets = load(os.path.join(restuldir,roi,"rotated_datasets.pkl"))
            subXs, subdfs = rotated_datasets["unrotated_Xs"], rotated_datasets["unrotated_dfs"]
            shufflelzerX = rotated_datasets["shuffled_lzerX"]
            rotated_lzerX, rotated_shufflelzerX, rotated_lzerdfs = rotated_datasets["rotated_lzerX"], rotated_datasets["rotated_shufflelzerX"], rotated_datasets["rotated_lzerdfs"]

            # run the analysis for regular unrotated activity patterns
            run_regs = ["train2test","loc2test","loc2train"]
            unrotated_reg_coefs, unrotated_results = RPCV_run_group(subXs,subdfs,subid_list,run_regs)    
            unrotated_results = unrotated_results.assign(roi=roi,dataset="orginal")
            unrotated_results.to_csv(os.path.join(restuldir,roi,"unrotated_results.csv"),index=False)
            dump(unrotated_reg_coefs,os.path.join(restuldir,roi,"unrotated_reg_coefs.pkl"))

            # run the analysis for rotated activity patterns
            rotatedlzer_results = []
            run_regs = ["loc2test","loc2train"]
            rotated_Xs = [[Xs[0],Xs[1],rXs] for Xs,rXs in zip(subXs,rotated_lzerX)]
            rotated_dfs =[[dfs[0],dfs[1], rdf] for dfs, rdf in zip(subdfs, rotated_lzerdfs)]
            rotatedlzer_reg_coefs, rotatedlzer_results = RPCV_run_group(rotated_Xs,rotated_dfs,subid_list,run_regs)    
            rotatedlzer_results = rotatedlzer_results.assign(roi=roi,dataset="rotated")
            rotatedlzer_results.to_csv(os.path.join(restuldir,roi,f"rotatedlzer_results.csv"),index=False)
            dump(rotatedlzer_reg_coefs,os.path.join(restuldir,roi,"rotatedlzer_reg_coefs.pkl"))

            # run the nalysis for shuffled activity patterns 
            shuffledlzer_results = []
            run_regs = ["loc2test","loc2train"]
            shulffledoutputdir = os.path.join(restuldir,roi,"shuffledlzer")
            checkdir(shulffledoutputdir)
            #shufflelzerX  is a list of nsub, in each element is a list of n_perm shuffled lzer activity patterns  
            # each shuffled lzer activity patterns is the same size as the original lzerX(only one run), 
            # therefore we need to copy it with np.vstack before applying it to odd/even naviX
            shuffle_Xs = [ [ [Xs[0],Xs[1],np.vstack([sXs[p],sXs[p]])] for Xs,sXs in zip(subXs,shufflelzerX)] for p in range(n_perm)]
            dfs =[ [dfs[0],dfs[1], rdf] for dfs, rdf in zip(subdfs, rotated_lzerdfs)]
            parallel(delayed(runrsaforshuffledlzer)(p,Xs,dfs,subid_list,run_regs,shulffledoutputdir) for p,Xs in zip(range(n_perm),shuffle_Xs))


            # run the analysis for rotated shuffled activity patterns    
            run_regs = ["loc2test","loc2train"]
            rotatedshulffledoutputdir = os.path.join(restuldir,roi,"rotatedshuffledlzer")
            checkdir(shulffledoutputdir)
            #rotated_shufflelzerX  is a list of nsub, in each element is a list of n_perm shuffled then rotated lzer activity patterns    
            rotatedshuffle_Xs = [ [ [Xs[0],Xs[1],srXs[p]] for Xs,srXs in zip(subXs,rotated_shufflelzerX)] for p in range(n_perm)]
            dfs =[ [dfs[0],dfs[1], rdf] for dfs, rdf in zip(subdfs, rotated_lzerdfs)]
            parallel(delayed(runrsaforshuffledlzer)(p,Xs,dfs,subid_list,run_regs,rotatedshulffledoutputdir) for p,Xs in zip(range(n_perm),rotatedshuffle_Xs))

            # without parallel
            #rotatedshuffledlzer_reg_coefs[roi], resultdf = RPCV_run_group(rotatedshuffle_Xs[p],dfs,subid_list,run_regs)    
            #curr_perm_res = resultdf.assign(roi=roi,dataset="rotatedshuffle",perm=p)
            #rotatedshuffledlzer_results.append(curr_perm_res)
            #curr_perm_res.to_csv(os.path.join(restuldir,"rotatedshuffledlzer",f"perm{p}_results.csv"),index=False)


# ## plot the results
resoverview_dir = os.path.join(restuldir,"resoverview")
checkdir(resoverview_dir)    

resoverview_dfs = []
for roi in rois:
    print(roi,end="\n")
    unrotated_resultsdf = pd.read_csv(os.path.join(restuldir,roi,"unrotated_results.csv"))
    rotatedlzer_resultsdf = pd.read_csv(os.path.join(restuldir,roi,"rotatedlzer_results.csv"))
    rs_dfs = []
    for p in range(n_perm):
        rs_dfs.append(pd.read_csv(os.path.join(restuldir,roi,"rotatedshuffledlzer",f"perm{p}_results.csv")))
    rotatedshuffledlzer_results_df = pd.concat(rs_dfs).reset_index(drop=True)
    rotatedshuffledlzer_results_sum = rotatedshuffledlzer_results_df.groupby(["dataset","roi","subid","metric"])["value"].mean().reset_index()
    #rotatedshuffledlzer_results_sum["subid"] = rotatedshuffledlzer_results_sum["perm"]
    resoverview_dfs.append(
        pd.concat(
            [unrotated_resultsdf.assign(perm=np.nan),
            rotatedlzer_resultsdf.assign(perm=np.nan),
            rotatedshuffledlzer_results_df
            ]
        ).assign(roi=roi).reset_index(drop=True)
    )

    retrievalcvdf = pd.concat(
        [unrotated_resultsdf.assign(perm=np.nan),
        rotatedlzer_resultsdf.assign(perm=np.nan),
        rotatedshuffledlzer_results_sum
        ]
        ).reset_index(drop=True)
    retrievalcvdf[["analysis","metricname"]] = retrievalcvdf.metric.str.split("-",expand=True)
    retrievalcvdf["subgroup"] = pd.Categorical(["G" if subid in generalizers else "nG" for subid in retrievalcvdf.subid],categories=["G","nG"],ordered=True)

    checkmetrics = ["eval_corr","eval_r2","compoweight","meanweightdiff","reg_percept","reg_distmod"]
    lzerpred_df = retrievalcvdf[(retrievalcvdf.analysis=="loc2test")&(retrievalcvdf.metricname.isin(checkmetrics))].copy()
    lzerpred_df.metricname = pd.Categorical(lzerpred_df.metricname, categories=checkmetrics,ordered=True)
    xvar, yvar, huevar = "subgroup", "value", "dataset"
    palettes = dict(zip(lzerpred_df[huevar].unique(),
                        sns.color_palette(None,lzerpred_df[huevar].nunique())
                        ))
    gs = sns.catplot(data = lzerpred_df[lzerpred_df.dataset!="orginal"].copy(),
                    x=xvar,y=yvar,hue=huevar,palette = palettes, sharex=False,sharey=False,
                    col="metricname",col_wrap=2, 
                    kind="violin",split=True,fill=False).set_titles("{col_name}")
    gs.map_dataframe(sns.stripplot,x=xvar,y=yvar,hue=huevar,palette = palettes,dodge=True)
    gs.fig.suptitle(f"{roi}: Localizer to Test Retrieval Performance",y=1.05)
    for gname,gdf in lzerpred_df[lzerpred_df.dataset!="orginal"].copy().groupby(["metricname","dataset","subgroup"]):
        print(gname)
        print(scipy.stats.ttest_1samp(gdf.value.values,0))
    gs.savefig(os.path.join(restuldir,"resoverview",f"{roi}_loc2test.png"))

    lzerpred_df = retrievalcvdf[(retrievalcvdf.analysis!="loc2train")&(retrievalcvdf.metricname.isin(checkmetrics))].copy()
    lzerpred_df.metricname = pd.Categorical(lzerpred_df.metricname, categories=checkmetrics,ordered=True)
    xvar, yvar, huevar = "subgroup", "value", "dataset"
    palettes = dict(zip(lzerpred_df[huevar].unique(),
                        sns.color_palette(None,lzerpred_df[huevar].nunique())
                        ))
    gs = sns.catplot(data = lzerpred_df[lzerpred_df.dataset!="orginal"].copy(),
                    x=xvar,y=yvar,hue=huevar,palette = palettes,
                    col="metricname",col_wrap=2, sharex=False,sharey=False,
                    kind="violin",split=True,fill=False).set_titles("{col_name}")
    gs.map_dataframe(sns.stripplot,x=xvar,y=yvar,hue=huevar,palette = palettes,dodge=True)
    gs.fig.suptitle(f"{roi}: Localizer to Train Retrieval Performance",y=1.05)
    gs.savefig(os.path.join(restuldir,"resoverview",f"{roi}_loc2train.png"))


unrotatedreg_coefs = dict(zip(rois, [{} for _ in rois]))
rotatedreg_coefs = dict(zip(rois, [{} for _ in rois]))
for roi in rois:
    rotatedreg_coefs[roi]   = load(os.path.join(restuldir,roi,"rotatedlzer_reg_coefs.pkl"))
    unrotatedreg_coefs[roi] = load(os.path.join(restuldir,roi,"unrotated_reg_coefs.pkl"))


dump(
    {
       "results": pd.concat(resoverview_dfs).reset_index(drop=True),
       "unrotated_coefs": unrotatedreg_coefs,
       "rotated_coefs":rotatedreg_coefs
    },
    os.path.join(resoverview_dir,"retrievalpatternCV_results.pkl")   
)



# """ Illustration of Rotation"""
# def get_rotation_matrix_from_angle(theta):
#     R = np.array([[np.cos(theta), -np.sin(theta)], 
#                   [np.sin(theta), np.cos(theta)]]
#                   )
#     return R

# ori_data = np.array([[2,0],
#                      [0,1],
#                      [0,0]])+1

# rotated_data = ori_data@get_rotation_matrix_from_angle(np.pi/4) - 4

# ori_data = ori_data+3

# plt.subplot()
# plt.scatter(ori_data[:,0],ori_data[:,1],label="original")
# plt.scatter(rotated_data[:,0],rotated_data[:,1],label="rotated")
# plt.axhline(0,color="black")
# plt.axvline(0,color="black")