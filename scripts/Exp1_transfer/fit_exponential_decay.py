import pandas as pd
import numpy as np
import os
import scipy
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import sys
project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(project_path)
from src.utils.composition_modelfit import multi_start_optimisation

data_path = r'E:\pirate_fmri\Analysis\data\Exp1_transfer'
data = pd.read_csv(os.path.join(data_path,f'treasurehuntdatawithprob.csv'))
## fit exponential decay curve to describe training process
def exp_decay(x,N0,rate):
    return N0*np.exp(-rate*x)

def fit_decay(fit_x,fit_y):
    #small_intercept=0.01 # this is to avoid log(0)
    #m0 = scipy.stats.linregress(fit_x,np.log(fit_y+small_intercept))
    #init_guess =  [np.exp(m0.intercept)-small_intercept, -m0.slope]
    #fitted_param, _ = curve_fit(f=exp_decay,xdata=fit_x,ydata=fit_y,p0=init_guess,maxfev=5000)

    obj_func = lambda params,xdata,ydata: np.sum((exp_decay(xdata,*params)-ydata)**2)
    fitted_param, _ = multi_start_optimisation(objective_func=obj_func,
                             bounds=[(0,5),(0,10)],
                             args=(fit_x,fit_y),
                             nstart=200,
                             n_jobs=10)
    pred_y = exp_decay(fit_x,*fitted_param)
    r2 = r2_score(fit_y,pred_y)
    return fitted_param, r2, pred_y

fitted_param_dfs = []
yvars = ["resp_dist", 'error_x', 'error_y']
for gnames,gdf in data.groupby(["subid","expt_map","stim_group"]):
    

    gdf_sum = gdf.groupby(["expt_map","cycle"])[yvars].mean().reset_index()
    gdf["cycle_within_map"] = [c if m=="training map" else c-6 for c,m in gdf[["cycle","expt_map"]].to_numpy()]
    gdf_sum["cycle_within_map"] = [c if m=="training map" else c-6 for c,m in gdf_sum[["cycle","expt_map"]].to_numpy()]
    
    if gnames[2] == "training map":
        gdf.sort_values(by="expt_index",inplace=True)
        gdf["trial_id"] = np.arange(gdf.shape[0])
    else:
        gdf["trial_id"] = gdf["cycle_within_map"]*9
    
    ops = [fit_decay(gdf_sum["cycle_within_map"].to_numpy(), gdf_sum[yvar].to_numpy()) for yvar in yvars]
    #ops = [fit_decay(gdf["trial_id"].to_numpy(), gdf[yvar].to_numpy()) for yvar in yvars]
    fitted_params, r2s, _ = zip(*ops)
    if np.logical_and(min(r2s)<0.3, gdf.isgeneralizer.values[0] == "generalizer"):
        print(gnames, fitted_params)
    tmpdf = pd.DataFrame(np.hstack([fitted_params,np.atleast_2d(r2s).T]),
                columns=["N0","rate","r2"])
    tmpdf["yvar"] = yvars
    tmpdf = tmpdf.assign(
                    subid = gnames[0],
                    expt_map = gnames[1],
                    stim_group = gnames[2]
                    )
                    
    fitted_param_dfs.append(tmpdf)
            
fitted_param_df = pd.concat(fitted_param_dfs,axis=0).reset_index(drop=True)
fitted_param_df.to_csv(os.path.join(data_path,"exponentialdecay_fittedparam.csv"))