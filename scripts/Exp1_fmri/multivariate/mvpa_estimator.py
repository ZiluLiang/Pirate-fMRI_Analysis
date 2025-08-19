"""
This module contains RSA estimator specific to the pirate
"""
from zpyhelper.MVPA.estimators import MetaEstimator
from zpyhelper.MVPA.preprocessors import split_data,scale_feature,concat_data
import numpy
from pingouin import partial_corr
from typing import Union
import pandas
import sys
import os
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path))


class CompositionalRetrieval_CV(MetaEstimator):
    def __init__(self,activitypatterns:list,
                 stim_dfs:list,
                 run_reg=None) -> None:
        
        #get the preprocessed and reordered data
        assert len(activitypatterns) == 3, f"expected length = 3, got {len(activitypatterns)}"
        assert len(stim_dfs) == 3, f"expected length = 3, got {len(stim_dfs)}"
        [navitrain_X,navitest_X,lzer_X], [navitrain_df,navitest_df,lzer_df] = activitypatterns, stim_dfs
        
        self.Xs = {
            "trainstim": navitrain_X,
            "teststim":  navitest_X
        }

        self.dfs = {
            "trainstim": navitrain_df,
            "teststim":  navitest_df
        }

        self.regression_configs = {
            "train2test": ["trainstim","teststim"] # in the order of independent var, dependent var
        }
        
        if run_reg is None:
            run_reg = list(self.regression_configs.keys())
        else:
            run_reg = [x for x in run_reg if x in self.regression_configs.keys()]
        assert len(run_reg)>0, "No valid regression configurations specified"
        self.run_reg = run_reg

    def data_OEsplitter(self,stimdf):
        """generate splitted based on stimuli dataframe to split data into odd and even splits

        Parameters
        ----------
        stimdf : pd.DataFrame
            stimuli dataframe

        Returns
        -------
        tuple
            a tuple of filers [oddrunsfilter, evenrunsfilter]
        """
           
        assert numpy.logical_and(stimdf.stim_task.nunique()==1,stimdf.stim_task.unique()[0]==0), f"must be navigation task(0), got {stimdf.stim_task.unique()} instead"
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
            
            conf_odd_fil, conf_even_fil = self.data_OEsplitter(configural_df)
            comp_odd_fil, comp_even_fil = self.data_OEsplitter(component_df)
                
            splitter = {"O2E": [[conf_odd_fil,comp_odd_fil],   [conf_even_fil,comp_even_fil]],
                        "E2O": [[conf_even_fil,comp_even_fil], [conf_odd_fil,comp_odd_fil]]
                        }

            # get the empirical retrieval patterns via cross-validation
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
                assert numpy.isnan(configural_X).sum() == 0, f"{regname} configural_X has nan values {configural_X}"
                assert numpy.isnan(component_X).sum() == 0, f"{regname} component_X has nan values {component_X}"
                
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
                reg_res[f"fit_corr_{cvsplit}"] = numpy.mean([pearsonr(tarpat,predpat).statistic for tarpat,predpat in zip(fit_tar.T,fit_pred.T)]) 
                reg_res[f"eval_corr_{cvsplit}"] = numpy.mean([pearsonr(tarpat,predpat).statistic for tarpat,predpat in zip(eval_tar.T,eval_pred.T)]) 
                
            for metric in ["r2","corr"]:
                reg_res[f"fit_{metric}"]  = numpy.mean([reg_res[f"fit_{metric}_{cvsplit}"] for cvsplit in splitter.keys()])
                reg_res[f"eval_{metric}"] = numpy.mean([reg_res[f"eval_{metric}_{cvsplit}"] for cvsplit in splitter.keys()])            

            reg_coefs = numpy.mean(cv_split_reg_coefs[regname],axis=0)
            for cvsplit in splitter.keys():
                reg_res.pop(f"fit_r2_{cvsplit}")
                reg_res.pop(f"eval_r2_{cvsplit}")
            
            
            component_df  = component_df[comp_odd_fil].copy()
            configural_df = configural_df[conf_odd_fil].copy()
            
            # model retrieval patterns: highD-perceptual matching; lowD-distance modulated
            percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                    component_df[["stim_x","stim_y"]].to_numpy(),
                                    lambda u,v: sum(u==v))
                
            compo_locrep  = component_df[["stim_x","stim_y"]].to_numpy()*2
            config_locrep = configural_df[["stim_x","stim_y"]].to_numpy()*2
            loc_reg = LinearRegression(fit_intercept=False,positive=False).fit(compo_locrep.T, config_locrep.T)
            dist_mod_reg_coefs = loc_reg.coef_
            
            # regress the empirical on the models
            pdcoef = LinearRegression(fit_intercept=True).fit(
                        X=numpy.array([scale_feature(dist_mod_reg_coefs.flatten(),2),scale_feature(percept_reg_coefs.flatten(),2)]).T,
                        y=scale_feature(reg_coefs.flatten(),2)
                    ).coef_

            regcoefmats[regname] = [reg_coefs,percept_reg_coefs,dist_mod_reg_coefs]
            pred_configurals[regname] = numpy.mean(cv_split_preds,axis=0)
            

            n_rel = 16*2 # 16 test stimuli, each has 2 relevant training stimuli
            compo_stims = reg_coefs[percept_reg_coefs==1]
            compo_stims_x = reg_coefs[:,:4][percept_reg_coefs[:,:4]==1]
            compo_stims_y = reg_coefs[:,4:][percept_reg_coefs[:,4:]==1]
            
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
                         compo_stims.mean()-noncompo_stims.mean(), compo_stims_x.mean()-noncompo_stims_x.mean(), compo_stims_y.mean()-noncompo_stims_y.mean()]
            prim_names = ["reg_distmod","reg_percept",
                          "compoweight_x","compoweight_y","compoweight", 
                          "noncompoweight_x", "noncompoweight_y","noncompoweight",
                          "meanweightdiff", "meanweightdiff_x","meanweightdiff_y"]
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