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
    def __init__(self,activitypattern:numpy.ndarray, stim_df:pandas.DataFrame) -> None:
        
        activitypattern  = numpy.atleast_2d(activitypattern)
        
        # make sure that required columns are there in the dataframe
        check_cols = ["stim_x","stim_y","stim_id","stim_group", "stim_session","training_axset", "training_axlocTL", "taskname"]
        assert all([x in stim_df.columns for x in check_cols])
        assert activitypattern.shape[0] == stim_df.shape[0], f"activitypattern shape {activitypattern.shape} does not match stim_df shape {stim_df.shape}"
        
        # make sure that all 4 runs of treasurehunt task and 1 run of localizer task are in the data
        assert stim_df.stim_session.nunique() == 5 
        assert stim_df.taskname.nunique() ==2 
        
        #get the preprocessed and reordered data
        [navitrain_X,navitest_X,lzer_X], [navitrain_df,navitest_df,lzer_df] = self._average_and_reorder(activitypattern,stim_df)
        
        self.Xs = {
            "trainstim": navitrain_X,
            "teststim":  navitest_X,
            "locations": lzer_X
        }

        self.dfs = {
            "trainstim": navitrain_df,
            "teststim":  navitest_df,
            "locations": lzer_df
        }

        self.regression_configs = {
            "train2test": ["trainstim","teststim"], # in the order of independent var, dependent var
            "loc2test": ["locations","teststim"],
            "loc2train": ["locations","trainstim"]
        }
    
    def _average_and_reorder(self,activitypattern,stimdf):
        
        # filter out data from treasure hunt task
        navi_filter = numpy.vstack(
            [stimdf.taskname.to_numpy() == "navigation",
            [not numpy.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]] #filter out central stimulus
        ).all(axis=0)
        
        navi_X = activitypattern[navi_filter,:]
        assert navi_X.shape[0] == 24*4 # because we are ignoring the central sitmuli here, so we have (25-1)*4 stimuli in total
        navi_df = stimdf[navi_filter].copy().reset_index(drop=True)

        #then we process the treasure hunt task data here: we get the average of odd and even splits
        navi_Xsess = split_data(navi_X,navi_df.stim_session.to_numpy())
        oddX,evenX = numpy.mean([navi_Xsess[0],navi_Xsess[2]],axis=0), numpy.mean([navi_Xsess[1],navi_Xsess[3]],axis=0)
        # then we z-score for each split
        oddX,evenX = scale_feature(oddX,2), scale_feature(evenX,2)
        # and we put them back into a matrix
        navi_X = concat_data([oddX,
                              evenX
                             ])
        navi_df = navi_df[navi_df.stim_session<2].copy().reset_index(drop=True)

        ## split into training and test for reordering: this is to make sure the final weight matrices are ordered in the way we want
        training_filter = navi_df.stim_group==1
        test_filter     = navi_df.stim_group==0
        ### reorder training stimuli
        trdf = navi_df[training_filter].copy().reset_index(drop=True)
        trX  = navi_X[training_filter]
        trdfneworder = trdf.sort_values(by=['stim_session','training_axset','training_axlocTL'])
        new_order    = trdfneworder.index
        training_df  = trdfneworder.reset_index(drop=True).assign(stim_task=0)
        training_X   = trX[new_order,:]
        ### reorder test stimuli
        teX  = navi_X[test_filter]
        tedf = navi_df[test_filter].copy().reset_index(drop=True)
        teneworder    = tedf.sort_values(by=['stim_session','stim_x','stim_y'])
        teneworderidx = teneworder.index
        test_X        = teX[teneworderidx,:]
        test_df       = teneworder.reset_index(drop=True).assign(stim_task=0)

        # filter out data from localizer task
        lzer_filter = numpy.vstack(
            [stimdf.taskname.to_numpy() == "localizer",
            [not numpy.logical_and(x==0,y==0) for x,y in stimdf[["stim_x","stim_y"]].to_numpy()]]#filter out central stimulus
        ).all(axis=0)
        lzerX  = activitypattern[lzer_filter,:]
        # then we z-score for each split
        lzerX  = scale_feature(lzerX,2)
        # reoder localizer stimuli 
        lzdfneworder = stimdf[lzer_filter].copy().reset_index(drop=True).sort_values(by=['stim_session','training_axset','training_axlocTL'])
        new_order    = lzdfneworder.index
        lzer_df      = lzdfneworder.reset_index(drop=True).assign(stim_task=1)
        lzer_X       = lzerX[new_order,:]

        return [training_X,test_X,lzer_X],  [training_df,test_df,lzer_df]


    def _navidata_OEsplitter(self,stimdf,activitypattern):
        """
        generate filters to split data into odd and even splits
        """
        # makre stimdf and activitypattern have same number of rows
        assert activitypattern.shape[0] == stimdf.shape[0]
        # make sure we are splitting the treasure hunt task data
        assert numpy.logical_and(stimdf.stim_task.nunique()==1,stimdf.stim_task.unique()[0]==0), f"must be navigation task(0), got {stimdf.stim_task.unique()} instead"
        # make sure odd and even runs are already averaged separately
        assert numpy.array_equal(stimdf.stim_session.unique(),[0,1]), f"got sessions {stimdf.stim_session.unique()}, expecting [0,1]"
        odd_f  = stimdf.stim_session.to_numpy() == 0
        even_f = stimdf.stim_session.to_numpy() == 1
        return odd_f,even_f

    def fit(self):
        result = []
        result_names = []
        regcoefmats = {}
        cv_split_reg_coefs = {}
        pred_configurals = {}
        for regname, rcfg in self.regression_configs.items():
            # get data
            component_X,  configural_X = self.Xs[rcfg[0]], self.Xs[rcfg[1]]
            component_df, configural_df = self.dfs[rcfg[0]], self.dfs[rcfg[1]]

            # split data for cross validation
            if regname=="train2test":
                # if train2test do for each odd-even split, and see if the results are consistent
                conf_odd_fil, conf_even_fil = self._navidata_OEsplitter(configural_df,configural_X)
                comp_odd_fil, comp_even_fil = self._navidata_OEsplitter(component_df,component_X)
            else:
                # if lzer2navi, only split for navigation data and see if results are consistent (because we only have one run of localizer data)
                conf_odd_fil, conf_even_fil = self._navidata_OEsplitter(configural_df,configural_X)
                comp_odd_fil, comp_even_fil = component_df.stim_task.values == 1,component_df.stim_task.values == 1

            splitter = {"O2E": [[conf_odd_fil,comp_odd_fil],   [conf_even_fil,comp_even_fil]],
                        "E2O": [[conf_even_fil,comp_even_fil], [conf_odd_fil,comp_odd_fil]]
                        }
                    
            # model fitting to obtain retrieval pattern matrix and model evaluation
            reg_res = {}
            cv_split_reg_coefs[regname] = []            
            cv_split_preds = []            
            for cvsplit,[[fitf_tar,fitf_X],[evalf_tar,evalf_X]] in splitter.items():
                fit_tar,  fit_X  = configural_X[fitf_tar,:].T, (component_X[fitf_X,:]).T
                eval_tar, eval_X = configural_X[evalf_tar,:].T, (component_X[evalf_X,:]).T
                # make sure there is no nan bc LinearRegression does not like it
                assert numpy.isnan(configural_X).sum() == 0, f"{regname} configural_X has nan values {configural_X}"
                assert numpy.isnan(component_X).sum() == 0, f"{regname} component_X has nan values {component_X}"
                ## standardize the features and targets for regression model
                fit_X, eval_X = scale_feature(fit_X,1), scale_feature(eval_X,1)
                fit_tar, eval_tar = scale_feature(fit_tar,1), scale_feature(eval_tar,1)
                # fit the weights on the fit set
                reg_estimator = LinearRegression(fit_intercept=True,positive=False).fit(fit_X,fit_tar)
                # predict on the fit and eval set
                fit_pred, eval_pred = reg_estimator.predict(fit_X), reg_estimator.predict(eval_X)
                # get the weights
                coef_mat = reg_estimator.coef_
                
                cv_split_preds.append(eval_pred)
                cv_split_reg_coefs[regname].append(scale_feature(coef_mat,2))
                reg_res[f"fit_r2_{cvsplit}"]  = r2_score(fit_tar,fit_pred)
                reg_res[f"eval_r2_{cvsplit}"] = r2_score(eval_tar,eval_pred)                
                reg_res[f"fit_corr_{cvsplit}"] = numpy.mean([pearsonr(tarpat,predpat).statistic for tarpat,predpat in zip(fit_tar.T,fit_pred.T)]) 
                reg_res[f"eval_corr_{cvsplit}"] = numpy.mean([pearsonr(tarpat,predpat).statistic for tarpat,predpat in zip(eval_tar.T,eval_pred.T)]) 
                reg_res[f"fit_zcorr_{cvsplit}"] = numpy.mean([numpy.arctanh(pearsonr(tarpat,predpat).statistic) for tarpat,predpat in zip(fit_tar.T,fit_pred.T)]) 
                reg_res[f"eval_zcorr_{cvsplit}"] = numpy.mean([numpy.arctanh(pearsonr(tarpat,predpat).statistic) for tarpat,predpat in zip(eval_tar.T,eval_pred.T)]) 
            
            for metric in ["r2","corr","zcorr"]:
                reg_res[f"fit_{metric}"]  = numpy.mean([reg_res[f"fit_{metric}_{cvsplit}"] for cvsplit in splitter.keys()])
                reg_res[f"eval_{metric}"] = numpy.mean([reg_res[f"eval_{metric}_{cvsplit}"] for cvsplit in splitter.keys()])            

            # average across splits and then store the retrieval pattern matrix
            reg_coefs = numpy.mean(cv_split_reg_coefs[regname],axis=0)
            for cvsplit in splitter.keys():
                for metric in ["r2","corr","zcorr"]:            
                    reg_res.pop(f"fit_{metric}_{cvsplit}")
                    reg_res.pop(f"eval_{metric}_{cvsplit}")
                
            # generate model retrieval pattern matrices
            component_df  = component_df[comp_odd_fil].copy()
            configural_df = configural_df[conf_odd_fil].copy()
            
            if regname=="loc2train":
                percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                      component_df[["stim_x","stim_y"]].to_numpy(),
                                      lambda u,v: numpy.array_equal(u,v))
            else:
                percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                        component_df[["stim_x","stim_y"]].to_numpy(),
                                        lambda u,v: sum(u==v))
                
            compo_locrep  = component_df[["stim_x","stim_y"]].to_numpy()*2
            config_locrep = configural_df[["stim_x","stim_y"]].to_numpy()*2
            loc_reg = LinearRegression(fit_intercept=False,positive=False).fit(compo_locrep.T, config_locrep.T)
            dist_mod_reg_coefs = loc_reg.coef_
            
            regcoefmats[regname] = [reg_coefs,percept_reg_coefs,dist_mod_reg_coefs]
            pred_configurals[regname] = numpy.mean(cv_split_preds,axis=0)
            
            pdcoef = LinearRegression(fit_intercept=True).fit(
                        X=scale_feature(numpy.array([dist_mod_reg_coefs.flatten(),percept_reg_coefs.flatten()]).T,1),
                        y=scale_feature(reg_coefs.flatten(),2)
                    ).coef_

            compo_stims = reg_coefs[percept_reg_coefs==1]
            noncompo_stims = reg_coefs[percept_reg_coefs==0]
            n_rel = 8 if regname == "loc2train" else 32
            assert compo_stims.size == n_rel, f"expected to have {n_rel} relevant stimuli, got {compo_stims.size} instead"
            assert noncompo_stims.size == (reg_coefs.size-n_rel), f"expected to have {reg_coefs.size-n_rel} irrelevant stimuli, got {noncompo_stims.size} instead"

            result = result + list(reg_res.values()) + \
                        [pdcoef[0], pdcoef[1], compo_stims.mean(), compo_stims.mean()-noncompo_stims.mean(), compo_stims.sum()-noncompo_stims.sum()]
            prim_names = ["reg_distmod","reg_percept","meancompoweight", "meanweightdiff", "sumweightdiff"]
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




class CompositionalRetrieval(MetaEstimator):
    def __init__(self,activitypattern:numpy.ndarray,
                 stim_df:pandas.DataFrame) -> None:
        activitypattern  = numpy.atleast_2d(activitypattern)
        check_cols = ["stim_x","stim_y","stim_id","stim_color","stim_shape", "stim_group","stim_task"]
        assert all([x in stim_df.columns for x in check_cols])
        assert activitypattern.shape[0] == stim_df.shape[0]
        
        #z-score the activity pattern independently for each task (because the scale of activity of localizer and pirate task can be very different)
        activitypattern = concat_data([scale_feature(x,s_dir=2) for x in split_data(activitypattern,stim_df.stim_task.to_numpy())])
        navitrain_filter = numpy.all(numpy.vstack([
            stim_df["stim_task"].to_numpy()==0,
            stim_df["stim_group"].to_numpy()==1,
        ]),axis=0)
        navitest_filter = numpy.all(numpy.vstack([
            stim_df["stim_task"].to_numpy()==0,
            stim_df["stim_group"].to_numpy()==0
        ]),axis=0)
        lzer_filter = numpy.all(numpy.vstack([
            stim_df["stim_task"].to_numpy()==1,
        ]),axis=0)
        
        navitrain_X, navitrain_df = activitypattern[navitrain_filter,:] , stim_df[navitrain_filter].copy().reset_index(drop=True)
        navitest_X,  navitest_df  = activitypattern[navitest_filter,:] ,  stim_df[navitest_filter].copy().reset_index(drop=True)
        lzer_X,      lzer_df      = activitypattern[lzer_filter,:] ,      stim_df[lzer_filter].copy().reset_index(drop=True)
        
        self.Xs = {
            "teststim":  navitest_X,
            "trainstim": navitrain_X,
            "locations": lzer_X
        }

        self.dfs = {
            "teststim":  navitest_df,
            "trainstim": navitrain_df,
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
            reg_estimator = LinearRegression().fit(component_X.T,configural_X.T)
            reg_r2 = reg_estimator.score(component_X.T,configural_X.T)
            reg_coefs = reg_estimator.coef_
            if regname=="loc2train":
                percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                      component_df[["stim_x","stim_y"]].to_numpy(),
                                      lambda u,v: numpy.array_equal(u,v))
                dist_mod_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                       component_df[["stim_x","stim_y"]].to_numpy(),
                                        lambda u,v: numpy.sqrt(2)-numpy.linalg.norm(u-v))
            else:
                percept_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                        component_df[["stim_x","stim_y"]].to_numpy(), 
                                        lambda u,v: sum(u==v)/2  # on/off only, no graded weight, divided by 2 so that the weights sums up to 1
                                        )
                dist_mod_reg_coefs = cdist(configural_df[["stim_x","stim_y"]].to_numpy(),
                                        component_df[["stim_x","stim_y"]].to_numpy(),
                                        lambda u,v: numpy.sum( (1-numpy.abs(u-v))*(1*(v!=0)) ) # Things that are too far away will be negatively weighted, here the weights should also sums up to 1
                                        )
            
            regcoefmats[regname] = [reg_coefs,percept_reg_coefs,dist_mod_reg_coefs]
            
            pdcoef = LinearRegression().fit(
                        X=scale_feature(numpy.array([dist_mod_reg_coefs.flatten(),percept_reg_coefs.flatten()]).T,1),
                        y=scale_feature(reg_coefs.flatten(),2)
                    ).coef_

            # similar metric but using partial rank correlation instead of regression
            corrdatadf = pandas.DataFrame({"regres":reg_coefs.flatten(),"percept":percept_reg_coefs.flatten(),"distmod":dist_mod_reg_coefs.flatten()})
            corr_percept = partial_corr(data=corrdatadf,x="regres",y="percept",x_covar="distmod",method="spearman")["r"][0]
            corr_dmod = partial_corr(data=corrdatadf,x="regres",y="distmod",x_covar="percept",method="spearman")["r"][0]
            zcorr_percept,zcorr_distmod = numpy.arctanh(corr_percept), numpy.arctanh(corr_dmod) #"zcorr_distmod","zcorr_percept",

            compo_stims = reg_coefs[percept_reg_coefs>0]
            noncompo_stims = reg_coefs[percept_reg_coefs==0]
            assert compo_stims.size>0
            assert noncompo_stims.size>0

            result = result + [reg_r2, pdcoef[1], pdcoef[0], zcorr_percept,zcorr_distmod, compo_stims.mean(), compo_stims.mean()-noncompo_stims.mean(), compo_stims.sum()-noncompo_stims.sum()]
            prim_names = ["comporegr2" , "reg_percept", "reg_distmod", "zcorr_percept", "zcorr_distmod" ,"meancompoweight", "meanweightdiff", "sumweightdiff"]
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