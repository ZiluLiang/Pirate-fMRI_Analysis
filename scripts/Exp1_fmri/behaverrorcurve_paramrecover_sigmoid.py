import scipy
from scipy.stats import norm,truncnorm,uniform

import json
import numpy as np

import pandas as pd
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os
import sys
from joblib import Parallel, delayed, cpu_count, dump,load
from zpyhelper.filesys import checkdir

project_path = r'E:\pirate_fmri\Analysis'
sys.path.append(project_path)

from src.utils.composition_modelfit import multi_start_optimisation,wise_start_optimisation, multi_start_presetgrid_optimisation

from src.utils.composition_modelfit import multi_start_optimisation,wise_start_optimisation, multi_start_presetgrid_optimisation

study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
studydata_dir  = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    fmribeh_dir = pirate_defaults['directory']['fmribehavior']
    fmridata_dir = pirate_defaults['directory']['fmri_data']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(len(subid_list))
print(len(cohort1ids))
print(len(cohort2ids))

from scipy.special import expit
def sigmoid(t,u,s,b):
    z = s*(t-b)
    err = u*expit(z)
    return err

# def errorcurve_ols_reducedparam(trueerr,t,s,b):
#     z = s*(t-b)
#     h = expit(z)
#     ls = np.dot(trueerr,trueerr) - np.dot(h,trueerr)/np.dot(h,h)
#     return ls

# def compute_carryingcapacity(trueerr,t,s,b):
#     z = s*(t-b)
#     h = expit(z)
#     u = np.dot(h,trueerr)/np.dot(h,h)
#     return u

def grab_init_guess_sigmoid(t,err):
    uni_t = np.unique(t)
    uni_err = np.array([err[t==ut].mean() for ut in uni_t])
    u = max(uni_err)
    diff_from_half = np.abs(uni_err-0.5*u)
    miderr_idx = np.where((diff_from_half - diff_from_half.min())<1e-3)
    b = np.mean(uni_t[miderr_idx])
    idx_post = [min(midx+1,len(uni_t)-1) for midx in miderr_idx[0]]
    idx_pre  = [max(midx-1,0) for midx in miderr_idx[0]]
    miderr_deriv = [(uni_err[ipo] - uni_err[ipe])/(uni_t[ipo] - uni_t[ipe]) for ipo,ipe in zip(idx_post,idx_pre)]
    s = 4*np.mean(miderr_deriv)
    p0s = [u,s,b]
    
    # param_bounds = [(-np.inf,np.inf) for _ in range(len(p0s))] if len(param_bounds)==0 else param_bounds
    # assert len(param_bounds) == len(p0s)
    # bounded_p0s = []
    # for p0,(pmin,pmax) in zip(p0s,param_bounds):
    #     if p0<pmin or p0>pmax:
    #         p0 = np.random.rand(1)[0]*(pmax-pmin)+pmin
    #     #p0 = min([p0,pmax])
    #     #p0 = max([p0,pmin])
    #     bounded_p0s.append(p0)    
    return p0s

def sample_paramgrid(mu,sig,bnd,N):
    if mu<bnd[0] or mu>bnd[1]:
        return sample_paramgrid_from_uniform(bnd,N)
    else:
        return sample_paramgrid_from_truncnorm(mu,sig,bnd,N)

def sample_paramgrid_from_truncnorm(mu,sig,bnd,N):
    # for truncated norm, the a,b parameter should be specified as follows:
    # a_trunc and b_trunc are the abscissae at which we wish to truncate the distribution
    # a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    a_trunc,b_trunc = bnd # truncate error at 0,1
    a, b = (a_trunc - mu) / sig, (b_trunc - mu) / sig
    qlist = np.arange(N+1)/N
    return [truncnorm.ppf(q,a,b,mu,sig) for q in qlist]

def sample_paramgrid_from_uniform(bnd,N):
    # for truncated norm, the a,b parameter should be specified as follows:
    # a_trunc and b_trunc are the abscissae at which we wish to truncate the distribution
    # a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    l,u = bnd # truncate error at 0,1
    s = u-l
    qlist = np.arange(N+1)/N
    return [uniform.ppf(q,l,s) for q in qlist]

param_bound = dict(zip(["u","s","b"],
                       [(1e-5,1),(-4,0),(1,12*9)] ))#bounds of param capacity, slope,inflection point

t = (np.arange(12*9)+1)
inflection_points = (np.arange(12)*9+1.5)
colors = sns.color_palette("viridis",len(inflection_points))
for b,lc in zip(inflection_points,colors):
    plt.plot(t,sigmoid(t,1,-4,b),color=lc,label='%.1f-%.2f'%(b,-4),linestyle="--")
    plt.plot(t,sigmoid(t,1,-8,b),  color=lc,label='s=%.1f,b=%.2f'%(b,-8),linestyle="-")
    plt.xticks(inflection_points-.5)
plt.legend(loc="center",bbox_to_anchor=(1.5,0.5),ncol=3)

# lets do some parameter recovery
### first we sample combinations of parameters with the constraint that l<u
noise_scale = .1 #0.000001
#simulated_param_bound = dict(zip(["u","s","bodd","beven"],
#                       [(0,1),(-4,0),(0,12*9)] ))
simulated_param_bound = param_bound
Ns = [11,11,11]
true_param_list = [np.linspace(start=pmin,stop=pmax,num=N,endpoint=False) for N,(pmin,pmax) in zip(Ns,simulated_param_bound.values())]
true_param_combs = np.array([pcomb for pcomb in itertools.product(*true_param_list)])
true_param_combs.shape
### simulate experiment 
Ts = (np.arange(12*9)+1) #+norm(0,noise_scale).rvs(np.size(Ts))
simulated_Es = [sigmoid(Ts,*pcomb) for pcomb in true_param_combs]
#T_odd = Ts[np.mod(Ts,2)==1]
#T_even = Ts[np.mod(Ts,2)==0]
#simulated_Es = [np.hstack(sigmoid(T_odd,pcomb[0],pcomb[1],pcomb[2]),sigmoid(T_odd,pcomb[0],pcomb[1],pcomb[3]))+norm(0,noise_scale).rvs(np.size(Ts)) for pcomb in true_param_combs]

p0df_list = []
for k,(truepcomb,ydata,p0s) in enumerate(zip(true_param_combs,simulated_Es,true_param_combs)):
    xdata = Ts
    print(f"Recovering param comb {k}/{len(true_param_combs)}:  {truepcomb}")
    objective_fun = lambda params: np.sum((ydata - sigmoid(xdata,*params))**2)
    
    p0 = grab_init_guess_sigmoid(Ts,ydata) #,param_bounds=list(param_bound.values())
    param_bound["u"] = (max(0,p0[0]-3*np.std(ydata)/np.sqrt(ydata.size)),
                        min(p0[0]+3*np.std(ydata)/np.sqrt(ydata.size),1))
    
    pred_err = sigmoid(xdata,*p0)
    r2_predtrue = scipy.stats.linregress(x=ydata,y=pred_err).rvalue**2

    p0df = pd.DataFrame()
    p0df["param"] = list(param_bound.keys())
    p0df["groundtruth"] = truepcomb
    p0df["initguess"] = p0
    p0df = p0df.assign(param_comb_id=k,r2=r2_predtrue)

    p0df_list.append(p0df)
    
p0_df_long = pd.concat(p0df_list,axis=0).reset_index(drop=True)
p0_df_long["groundtruth"] = p0_df_long["groundtruth"].round(2)
#p0_df_all.to_csv(os.path.join(studydata_dir,"pretrainerror_paramrecovery.csv"))
p0_df_wide = p0_df_long.pivot(index=["param_comb_id","r2"],
                columns="param",
                values=["groundtruth","initguess"]).reset_index()
p0_df_wide = p0_df_wide.set_axis(p0_df_wide.columns.map(''.join), axis=1)
# First let's check the quality of initial guess
plt.subplots(1,len(param_bound),figsize=(4*len(param_bound),3))
for j,(param_name,pdf) in enumerate(p0_df_long.groupby("param")):
    plt.subplot(1,len(param_bound),j+1)
    sns.scatterplot(p0_df_wide,
                    x=f"groundtruth{param_name}",y=f"initguess{param_name}",
                    hue="groundtruthb")
    plt.plot(pdf["groundtruth"],pdf["groundtruth"],color="darkgrey")
    plt.title(param_name)
plt.tight_layout()


plt.subplots(1,2,figsize=(4*2,4))
plt.subplot(1,2,1)
sns.heatmap(p0_df_wide.pivot_table(index='groundtruthb', columns='groundtruthu', values='r2',aggfunc="mean"))
plt.xlabel("u")
plt.ylabel("b")
plt.title("R2 ~ b and u")
plt.subplot(1,2,2)
sns.heatmap(p0_df_wide.pivot_table(index='groundtruthb', columns='groundtruths', values='r2',aggfunc="mean"))
plt.xlabel("s")
plt.ylabel("b")
plt.title("R2 ~ b and s")
plt.tight_layout()

simresdf_list = []
for k,(truepcomb,ydata,p0s) in enumerate(zip(true_param_combs,simulated_Es,true_param_combs)):
    xdata = Ts
    print(f"Recovering param comb {k}/{len(true_param_combs)}:  {truepcomb}")
    objective_fun = lambda params: np.sum((ydata - sigmoid(xdata,*params))**2)
    
    p0 = grab_init_guess_sigmoid(Ts,ydata) #,param_bounds=list(param_bound.values())
    param_bound["u"] = (max(0,p0[0]-3*np.std(ydata)/np.sqrt(ydata.size)),
                        min(p0[0]+3*np.std(ydata)/np.sqrt(ydata.size),1))
    
    grid_edges = [sample_paramgrid(mu,(bnd[1]-bnd[0])/12,bnd,N=10) for mu,bnd in zip(p0,param_bound.values())]
    grid_lbs = [g[:-1] for g in grid_edges]
    grid_ubs = [g[1:] for g in grid_edges]
    x0s = [[np.random.uniform(low=glb,high=gub) for glb,gub in zip(pglb,pgub)] for pglb,pgub in zip(grid_lbs,grid_ubs)]
    x0_grids = list(itertools.product(*x0s))

    optimal_solution, optimal_fval = multi_start_presetgrid_optimisation(objective_func=objective_fun,
                        bounds=list(param_bound.values()),
                        preset_x0_grids=x0_grids,
                        n_jobs=10)
    print(f"           Recovery found: {optimal_solution}")

    pred_err = sigmoid(xdata,*optimal_solution)
    r2_predtrue = scipy.stats.linregress(x=ydata,y=pred_err).rvalue**2

    simresdf = pd.DataFrame()
    simresdf["param"] = list(param_bound.keys())
    simresdf["groundtruth"] = truepcomb
    simresdf["estimates"] = list(optimal_solution)
    simresdf["initguess"] = p0
    simresdf = simresdf.assign(param_comb_id=k,r2=r2_predtrue)

    simresdf_list.append(simresdf)
    
simres_df_all = pd.concat(simresdf_list,axis=0).reset_index(drop=True)
simres_df_all.to_csv(os.path.join(studydata_dir,"pretrainerror_paramrecovery.csv"))

