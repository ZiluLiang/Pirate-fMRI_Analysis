"""
This module contains RSA estimator specific to the pirate
"""
from zpyhelper.MVPA.estimators import MetaEstimator, PatternCorrelation, MultipleRDMRegression
from zpyhelper.MVPA.rdm import compute_rdm,compute_rdm_identity,compute_rdm_residual, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature
import numpy
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import itertools
import sys
import os
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(os.path.join(project_path))
from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM

from sklearn.linear_model import LinearRegression

class CompositionalRetrieval(MetaEstimator):
    def __init__(self,activitypattern:numpy.ndarray,
                 stim_df:pandas.DataFrame) -> None:
        activitypattern  = numpy.atleast_2d(activitypattern)
        check_cols = ["stim_x","stim_y","stim_id","stim_color","stim_shape", "stim_group","stim_task"]
        assert all([x in stim_df.columns for x in check_cols])
        assert activitypattern.shape[0] == stim_df.shape[0]
        
        noncenterfilter = [~numpy.logical_and(x==0,y==0) for x,y in stim_df[["stim_x","stim_y"]].to_numpy()]
        navitrain_filter = numpy.all(numpy.vstack([
            stim_df["stim_task"].to_numpy()==0,
            stim_df["stim_group"].to_numpy()==1,
            noncenterfilter
        ]),axis=0)
        navitest_filter = numpy.all(numpy.vstack([
            stim_df["stim_task"].to_numpy()==0,
            stim_df["stim_group"].to_numpy()==0
        ]),axis=0)
        lzer_filter = numpy.all(numpy.vstack([
            stim_df["stim_task"].to_numpy()==1,
            noncenterfilter
        ]),axis=0)
        
        navitrain_X, navitrain_df = activitypattern[navitrain_filter,:] , stim_df[navitrain_filter].copy().reset_index(drop=True)
        navitest_X,  navitest_df  = activitypattern[navitest_filter,:] ,  stim_df[navitest_filter].copy().reset_index(drop=True)
        lzer_X,      lzer_df       = activitypattern[lzer_filter,:] ,      stim_df[lzer_filter].copy().reset_index(drop=True)
        
        self.Xs = {
            "teststim":  navitrain_X,
            "trainstim": navitest_X,
            "locations": lzer_X
        }

        self.dfs = {
            "teststim":  navitrain_df,
            "trainstim": navitest_df,
            "locations": lzer_df
        }

        self.regression_configs = {
            "train2test": ["trainstim","teststim"], # in the order of independent var, dependent var
            "loc2test": ["locations","teststim"],
            "loc2train": ["locations","trainstim"]
        }

    def fit(self):
        result = []
        result_names = []
        regcoefmats = {}
        for regname, rcfg in self.regression_configs.items():
            component_X,  configural_X = self.Xs[rcfg[0]], self.Xs[rcfg[1]]
            component_df, configural_df = self.dfs[rcfg[0]], self.dfs[rcfg[1]]

            reg_coefs = LinearRegression().fit(component_X.T,configural_X.T).coef_
            true_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                   component_df[["stim_x","stim_y"]].to_numpy(),
                                   lambda u,v: sum(u==v))
            dist_mod_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                       component_df[["stim_x","stim_y"]].to_numpy(),
                                       lambda u,v: numpy.sum((2-numpy.abs(u-v))*(1*(v!=0))))
            regcoefmats[regname] = [reg_coefs,true_reg_coefs,dist_mod_reg_coefs]
            
            pdcoef = LinearRegression().fit(
                        X=scale_feature(numpy.array([dist_mod_reg_coefs.flatten(),true_reg_coefs.flatten()]).T,1),
                        y=scale_feature(reg_coefs.flatten(),2)
                    ).coef_
            corr_percept = spearmanr(reg_coefs.flatten(),true_reg_coefs.flatten()).statistic
            corr_dmod = spearmanr(reg_coefs.flatten(),dist_mod_reg_coefs.flatten()).statistic

            compo_stims = reg_coefs[true_reg_coefs==1]
            noncompo_stims = reg_coefs[true_reg_coefs==0]

            result = result + [pdcoef[0], pdcoef[1], numpy.arctanh(corr_dmod), numpy.arctanh(corr_percept), compo_stims.mean()-noncompo_stims.mean(), compo_stims.sum()-noncompo_stims.sum()]
            prim_names = ["reg_distmod","reg_percept","zcorr_distmod","zcorr_percept", "meanweightdiff", "sumweightdiff"]
            result_names = result_names + [f"{regname}-{x}" for x in prim_names]
            
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
    
