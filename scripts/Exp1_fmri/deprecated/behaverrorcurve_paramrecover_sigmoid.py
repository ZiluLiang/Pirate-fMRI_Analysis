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

from src.utils.composition_modelfit import multi_start_presetgrid_optimisation
def multi_start_presetgrid_optimisation(objective_func,bounds,preset_x0_grids=list,args=(),minimize_args={},
                                 n_jobs=1):
    rand_starts = preset_x0_grids
    # set up minimize function
    def_minimize_args = {}
    def_minimize_args.update(minimize_args)
    optim_func = lambda x0: minimize(objective_func,x0,args=args, bounds=bounds,**def_minimize_args)
    
    # multi-start grid search optimization
    with Parallel(n_jobs=n_jobs) as parallel:
        oplist = parallel(delayed(optim_func)(x0) for x0 in rand_starts)
    objfun_vals,solutions = zip(*[(res.fun,res.x) for res in oplist])
    optimal_solution      = solutions[numpy.argmin(objfun_vals)]
    optimal_fval          = objfun_vals[numpy.argmin(objfun_vals)]
    return optimal_solution, optimal_fval

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
def sample_value_from_truncnorm(mu,sig,bnd,size):
    a_trunc,b_trunc = bnd # truncate error at 0,1
    a, b = (a_trunc - mu) / sig, (b_trunc - mu) / sig
    return truncnorm.rvs(a,b,mu,sig,size)

def sigmoid(t,u,s,b):
    z = s*(t-b)
    err = u*expit(z)
    return err

def grab_init_guess_sigmoid(t,err):
    err = err + 1e-5
    uni_t = np.unique(t)
    uni_err = np.array([err[t==ut].mean() for ut in uni_t])
    u = 1.01*max(err)
    
    # guess b from midpoint
    diff_from_half = np.abs(uni_err-0.5*u)
    miderr_idx = np.where((diff_from_half - diff_from_half.min())<1e-3)
    b = np.mean(uni_t[miderr_idx])
   
    # guess s from slope around b
    #idx_prepost = np.where(np.logical_and(uni_t >= b-3,uni_t <= b+3))    
    #idx_post = [min(midx+1,len(uni_t)-1) for midx in miderr_idx[0]]
    #idx_pre  = np.where(uni_t == np.ceil(b))#[max(midx-1,0) for midx in miderr_idx[0]]
    #miderr_deriv  = [(uni_err[ipo] - uni_err[ipe])/(uni_t[ipo] - uni_t[ipe]) for ipo,ipe in zip(idx_post,idx_pre)]
    #miderr_deriv = -(np.max(uni_err[idx_prepost]) - np.min(uni_err[idx_prepost])) / (np.max(uni_t[idx_prepost])-np.min(uni_t[idx_prepost]))    
    #miderr_deriv =  scipy.stats.linregress(uni_t[idx_prepost], uni_err[idx_prepost]).slope
    #s = 4*miderr_deriv/u
    
    # guess s based on u and b 
    z = -np.log(u/err - 1)
    regres = scipy.stats.linregress((t-b), z)
    s = regres.slope

    p0s = [u,s,b]
    return p0s

def get_bounded_vals(p0s,param_bounds,flag_hitbound="random"):
    param_bounds = [(-np.inf,np.inf) for _ in range(len(p0s))] if len(param_bounds)==0 else param_bounds
    assert len(param_bounds) == len(p0s)
    bounded_p0s = []
    for p0,(pmin,pmax) in zip(p0s,param_bounds):
        if flag_hitbound == "random":
            if p0<pmin or p0>pmax:
                p0 = np.random.rand(1)[0]*(pmax-pmin)+pmin
        elif flag_hitbound == "bound":
            p0 = min([p0,pmax])
            p0 = max([p0,pmin])
        bounded_p0s.append(p0)  
    return bounded_p0s

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
                       [(1e-5,1),(-4,-1e-5),(1,12*9)] ))#bounds of param capacity, slope,inflection point


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
"""
#### for a pure sigmoid curve parameter recovery, we do the following:
simulated_param_bound = param_bound
Ns = [11,11,11]
true_param_list = [np.linspace(start=pmin,stop=pmax,num=N,endpoint=False) for N,(pmin,pmax) in zip(Ns,simulated_param_bound.values())]
true_param_combs = np.array([pcomb for pcomb in itertools.product(*true_param_list)])
true_param_combs.shape
##### simulate experiment 
Ts = (np.arange(12*9)+1) 
simulated_Es = [sigmoid(Ts,*pcomb) for pcomb in true_param_combs]
"""

#### for asymetric learning parameter recovery, we do the following:
N_sim = 1000
simulated_param_bound = param_bound
Ns = [100,100,100]
true_param_gridedgelist = [sample_paramgrid_from_uniform([pmin,pmax],N) for N,(pmin,pmax) in zip(Ns,simulated_param_bound.values())]
true_param_grid_lbs = [g[:-1] for g in true_param_gridedgelist]
true_param_grid_ubs = [g[1:] for g in true_param_gridedgelist]
true_param_list = [[np.random.uniform(low=glb,high=gub) for glb,gub in zip(pglb,pgub)] for pglb,pgub in zip(true_param_grid_lbs,true_param_grid_ubs)]
true_param_combs_all = np.array([pcomb for pcomb in itertools.product(*true_param_list)])
randidx = np.random.permutation(np.arange(true_param_combs_all.shape[0]))[:N_sim]
true_param_combs = true_param_combs_all[randidx]
true_param_combs.shape

### simulate experiment 
Ts = (np.arange(12*9)+1) 
T_odd = Ts[np.mod(Ts,2)==1]
T_even = Ts[np.mod(Ts,2)==0]
order_by_trial = np.argsort(np.concatenate([T_odd,T_even]))    
noise_levels={"low":0.05,"high":0.1} # 0.000001,"medium":
# to show effect of noise level
plt.subplots(1,3,figsize=(13,4))
for j, (nlvl,noise_scale) in enumerate(noise_levels.items()):
    plot_pcomb = true_param_combs[1]
    simulated_E_noiseless = np.hstack([sigmoid(T_odd,*plot_pcomb),sigmoid(T_even,*plot_pcomb)])
    simulated_E_noiselessordered = simulated_E_noiseless[order_by_trial]    
    simulated_E = np.array([sample_value_from_truncnorm(serr,noise_scale,[0,1],1) for serr in simulated_E_noiselessordered])
    plt.subplot(1,3,j+1)    
    plt.plot(Ts[np.mod(Ts,2)==1],simulated_E[np.mod(Ts,2)==1],label="odd trials")
    plt.plot(Ts[np.mod(Ts,2)==0],simulated_E[np.mod(Ts,2)==0],label="even trials")
    plt.xlabel("trial")
    plt.ylabel("simulated error")
    plt.title(f"{nlvl}: noise sigma ={noise_scale}")
    plt.legend()
    sns.move_legend(plt.gca(),loc="upper center",ncol=3)

Ngrids = dict(zip(["u","s","b"], [10,20,int(param_bound["b"][1]/4)]))
Ngrids = dict(zip(["u","s","b"], [5,10,int(param_bound["b"][1]/2)]))
simresdf_list = []
for nlvl,noise_scale in noise_levels.items():
    for k,truepcomb in enumerate(true_param_combs):
        print(f"{nlvl} Noise: Recovering param comb {k}/{len(true_param_combs)}:  {truepcomb}")
        simulated_Es_noiseless = np.hstack([sigmoid(T_odd,*truepcomb),sigmoid(T_even,*truepcomb)])
        simulated_Es_noiselessordered = simulated_Es_noiseless[order_by_trial]
        Eseq = np.array([sample_value_from_truncnorm(serr,noise_scale,[0,1],1)[0] for serr in simulated_Es_noiselessordered])
        
        #xdata = Ts
        #ydata = Eseq
        #tparam = truepcomb
        #dsname="all"
        T_odd = Ts[np.mod(Ts,2)==1]
        T_even = Ts[np.mod(Ts,2)==0]
        Eseq_odd = Eseq[np.mod(Ts,2)==1]
        Eseq_even = Eseq[np.mod(Ts,2)==0]
        tparam_odd = truepcomb
        tparam_even = truepcomb
        for dsname,xdata,ydata,tparam in zip(["odd","even"],[T_odd,T_even],[Eseq_odd,Eseq_even],[tparam_odd,tparam_even]):
            p0 = grab_init_guess_sigmoid(xdata,ydata)
            
            # try:
            #     # y_sliding_window = [ydata[max(j-8,0):min(j+8,len(ydata)-1)] for j in range(len(ydata))]
            #     # y_sliding_se = [ys.std()/np.sqrt(ys.size) for ys in y_sliding_window]
            #     # p0_new = scipy.optimize.curve_fit(sigmoid,xdata=xdata,ydata=ydata,
            #     #                                 p0=get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="bound"),
            #     #                                 sigma=y_sliding_se,
            #     #                                 max_nfev = 5000,
            #     #                                 bounds = ([bnd[0] for bnd in param_bound.values()],[bnd[1] for bnd in param_bound.values()]))
            #     # warmstart = p0_new[0]

            #     lsguess_objfun = lambda params: (ydata - sigmoid(xdata,*params))
            #     lsguess = scipy.optimize.least_squares(fun=lsguess_objfun,
            #                                 x0=get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="random"),
            #                                 bounds = ([bnd[0] for bnd in param_bound.values()],[bnd[1] for bnd in param_bound.values()]),
            #                                 loss="soft_l1")
            #     warmstart = lsguess.x
            # except:
            #     warmstart = p0
            
            # warmstartsigs = [(bnd[1]-bnd[0])/6 for bnd in param_bound.values()]
            
            # grid-search with warm start        
            #grid_edges = [sample_paramgrid(mu, sig, bnd, Ng) for mu,sig, bnd, Ng in zip(warmstart,warmstartsigs,param_bound.values(),Ngrids.values())]
            grid_edges = [sample_paramgrid_from_uniform(bnd, Ng) for bnd, Ng in zip(param_bound.values(),Ngrids.values())]
                
            grid_lbs = [g[:-1] for g in grid_edges]
            grid_ubs = [g[1:] for g in grid_edges]
            x0s = [[np.random.uniform(low=glb,high=gub) for glb,gub in zip(pglb,pgub)] for pglb,pgub in zip(grid_lbs,grid_ubs)]
            x0_grids = list(itertools.product(*x0s))
            ###### multi_start_presetgrid_optimisation calls `minimize` in scipy, so it requires an objective funtion that returns a scalar
            soft_l1 = lambda z: 2*((1+z)**0.5-1)
            cauchy = lambda z: np.log(1+z)
            gsmin_objfun_softl1 = lambda params: np.sum(soft_l1(((ydata - sigmoid(xdata,*params)))**2))
            gsmin_objfun_mse = lambda params: np.sum(((ydata - sigmoid(xdata,*params)))**2)
            
            optimal_solution, optimal_fval = multi_start_presetgrid_optimisation(objective_func=gsmin_objfun_mse,
                            bounds=list(param_bound.values()),
                            preset_x0_grids=x0_grids,
                            n_jobs=16)
            pred_err = sigmoid(xdata,*optimal_solution)
            r2_predtrue = scipy.stats.linregress(x=ydata,y=pred_err).rvalue**2
            r2_ceil = scipy.stats.linregress(Eseq_odd,Eseq_even).rvalue**2
            print(f"           Recovery found: {['%.3f'%x for x in optimal_solution]}, R2={'%.3f'%r2_predtrue}, R2ceil={'%.3f'%r2_ceil}, R2/ceil ={'%.3f'%(r2_predtrue/r2_ceil)}")

            simresdf = pd.DataFrame()
            simresdf["param"] = list(param_bound.keys())
            simresdf["groundtruth"] = tparam
            simresdf["estimates"] = list(optimal_solution)
            simresdf["initguess"] = p0
            simresdf = simresdf.assign(param_comb_id=k,r2=r2_predtrue,r2ceil = r2_ceil,datasplit=dsname,noiselevel=nlvl,noiselevel_num=noise_scale)

            simresdf_list.append(simresdf)
    
simres_df_all = pd.concat(simresdf_list,axis=0).reset_index(drop=True)
simres_df_all.to_csv(os.path.join(studydata_dir,"pretrainerror_paramrecovery_uniformgs_mseloss.csv"))


# ## to check the parameter recovery quality
# simres_df_all = pd.read_csv(os.path.join(studydata_dir,"pretrainerror_paramrecovery.csv"))
# p0_df_long = simres_df_all.copy()
# simres_df_all["groundtruth"] = simres_df_all["groundtruth"].round(2)
# p0_df_wide = simres_df_all.pivot(index=["param_comb_id","r2"],
#                 columns="param",
#                 values=["groundtruth","estimates"]).reset_index()
# p0_df_wide = p0_df_wide.set_axis(p0_df_wide.columns.map(''.join), axis=1)
# # First let's check the quality of initial guess
# plt.subplots(1,len(param_bound),figsize=(4*len(param_bound),3))
# hue_var = {"b":"u",
#            "s":"u",
#            "u":"b"}
# for j,(param_name,pdf) in enumerate(simres_df_all.groupby("param")):
#     plt.subplot(1,len(param_bound),j+1)
#     sns.scatterplot(p0_df_wide,
#                     x=f"groundtruth{param_name}",y=f"estimates{param_name}",
#                     hue=f"groundtruth{hue_var[param_name]}")
#     plt.plot(pdf["groundtruth"],pdf["groundtruth"],color="darkgrey")
#     plt.title(param_name)
# plt.tight_layout()


# plt.subplots(1,2,figsize=(4*2,4))
# plt.subplot(1,2,1)
# sns.heatmap(p0_df_wide.pivot_table(index='groundtruthb', columns='groundtruthu', values='r2',aggfunc="mean"))
# plt.xlabel("u")
# plt.ylabel("b")
# plt.title("R2 ~ b and u")
# plt.subplot(1,2,2)
# sns.heatmap(p0_df_wide.pivot_table(index='groundtruthb', columns='groundtruths', values='r2',aggfunc="mean"))
# plt.xlabel("s")
# plt.ylabel("b")
# plt.title("R2 ~ b and s")
# plt.tight_layout()

