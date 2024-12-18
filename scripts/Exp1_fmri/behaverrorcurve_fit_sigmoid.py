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
    idx_prepost = np.where(np.logical_and(uni_t >= b-3,uni_t <= b+3))
    #idx_post = [min(midx+1,len(uni_t)-1) for midx in miderr_idx[0]]
    #idx_pre  = np.where(uni_t == np.ceil(b))#[max(midx-1,0) for midx in miderr_idx[0]]
    #miderr_deriv  = [(uni_err[ipo] - uni_err[ipe])/(uni_t[ipo] - uni_t[ipe]) for ipo,ipe in zip(idx_post,idx_pre)]
    miderr_deriv = -(np.max(uni_err[idx_prepost]) - np.min(uni_err[idx_prepost])) / (np.max(uni_t[idx_prepost])-np.min(uni_t[idx_prepost]))
    s = 4*np.mean(miderr_deriv)/u
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
                       [(0,1),(-4,0),(1,12*9)] ))#bounds of param capacity, slope,inflection point

t = (np.arange(12*9)+1)
inflection_points = (np.arange(12)*9+1.5)
colors = sns.color_palette("viridis",len(inflection_points))
for b,lc in zip(inflection_points,colors):
    plt.plot(t,sigmoid(t,1,-4,b),color=lc,label='%.1f-%.2f'%(b,-4),linestyle="--")
    plt.plot(t,sigmoid(t,1,-8,b),  color=lc,label='s=%.1f,b=%.2f'%(b,-8),linestyle="-")
    plt.xticks(inflection_points-.5)
plt.legend(loc="center",bbox_to_anchor=(1.5,0.5),ncol=3)

runon_stage = "test"
doscale_x = False
error_data = pd.read_csv(os.path.join(studydata_dir,f"trialwisepretrain{runon_stage}data.csv")).drop(columns=["X","Unnamed..0"])
res_df_list = []
for subid, subdf in error_data.groupby("subid"):
    subg = "Generalizer" if subid in generalizers else "nonGeneralizer"
    subc = "First Cohort" if subid in cohort1ids else "Second Cohort"
    for yvar in ["error_x_rescaled","error_y_rescaled"]:
        print(subid,yvar)
        xvar = "trainingtrialid" if runon_stage =="train" else "testtrialid"
        xvar = f"{xvar}_scaled" if doscale_x else xvar
        odd_data = subdf[np.mod(subdf[xvar],2)==1].copy()
        even_data = subdf[np.mod(subdf[xvar],2)==0].copy()
        for dsname, ds in zip(["odd","even"],[odd_data,even_data]):
            xdata = ds[xvar].to_numpy()
            ydata = ds[yvar].to_numpy()

            # guess starting value for estimating the warm start for grid search            
            p0 = grab_init_guess_sigmoid(xdata,ydata)
            #print(f"{dsname} First Guess: {['%.3f'%x for x in p0]}")

            # get the warm start for grid search using nls
            y_sliding_window = [ydata[max(j-8,0):min(j+8,len(ydata)-1)] for j in range(len(ydata))]
            y_sliding_se = [ys.std()/np.sqrt(ys.size) for ys in y_sliding_window]
            p0_new = scipy.optimize.curve_fit(sigmoid,xdata=xdata,ydata=ydata,
                                              p0=get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="bound"),
                                              sigma=y_sliding_se,
                                              max_nfev = 5000,
                                              bounds = ([bnd[0] for bnd in param_bound.values()],[bnd[1] for bnd in param_bound.values()]))
            warmstart = p0_new[0]
            #print(f"{dsname} Second Guess: {['%.3f'%x for x in warmstart]}")
            #optimal_solution = warmstart
            
            
            # grid-search with smart start        
            u_bnd = (0,1)
            #param_bound["u"] = (max(0,warmstart[0]-(u_bnd[1]-u_bnd[0])/6),
            #                    min(warmstart[0]+(u_bnd[1]-u_bnd[0])/6,1))
            grid_edges = [sample_paramgrid(mu,(bnd[1]-bnd[0])/6,bnd,N=10) for mu,bnd in zip(warmstart,param_bound.values())]
            grid_lbs = [g[:-1] for g in grid_edges]
            grid_ubs = [g[1:] for g in grid_edges]
            x0s = [[np.random.uniform(low=glb,high=gub) for glb,gub in zip(pglb,pgub)] for pglb,pgub in zip(grid_lbs,grid_ubs)]
            x0_grids = list(itertools.product(*x0s))

            objective_fun = lambda params: np.sum(((ydata - sigmoid(xdata,*params)))**2)
            optimal_solution, optimal_fval = multi_start_presetgrid_optimisation(objective_func=objective_fun,
                              bounds=list(param_bound.values()),
                              preset_x0_grids=x0_grids,
                              n_jobs=16)

            print(f"{dsname} Estimation found: {['%.3f'%x for x in optimal_solution]}")
            pred_err = sigmoid(xdata,*optimal_solution)
            r2_predtrue = scipy.stats.linregress(x=ydata,y=pred_err).rvalue**2

            resdf = pd.DataFrame()
            resdf["param"] = list(param_bound.keys())
            resdf["estimates"] = list(optimal_solution)
            resdf = resdf.assign(subid=subid,subgroup=subg,subcohort=subc,datasplit=dsname,yvar=yvar,r2=r2_predtrue)

            res_df_list.append(resdf)

res_df_all = pd.concat(res_df_list,axis=0).reset_index(drop=True)
res_df_all.to_csv(os.path.join(studydata_dir,f"pretrain{runon_stage}errorfit_warmstartgridsearch.csv"))


#### If we want to plot the starting value probability
# def genpdist_paramgrid_from_truncnorm(mu,sig,bnd,N):
#     # for truncated norm, the a,b parameter should be specified as follows:
#     # a_trunc and b_trunc are the abscissae at which we wish to truncate the distribution
#     # a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
#     a_trunc,b_trunc = bnd # truncate error at 0,1
#     a, b = (a_trunc - mu) / sig, (b_trunc - mu) / sig
#     return truncnorm(a,b,mu,sig)
# p0rvs = [genpdist_paramgrid_from_truncnorm(mu,(bnd[1]-bnd[0])/6,bnd,N=50) for mu,bnd in zip(p0,param_bound.values())]
# nplot = len(p0rvs)
# plt.subplots(1,nplot,figsize=(4*nplot,4))
# for j, (param_name,pbound) in enumerate(param_bound.items()):
#     p0vals = np.linspace(pbound[0]*1.1,pbound[1]*1.1,num=200,endpoint=True)
#     plt.subplot(1,nplot,j+1)
#     plt.plot(p0vals,p0rvs[j].pdf(p0vals),color="black")
#     plt.axvline(p0[j],color="blue",label="init guess")
#     plt.axvline(pbound[0],color="grey",linestyle="dashed",label="param bound")
#     plt.axvline(pbound[1],color="grey",linestyle="dashed",label="param bound")
#     plt.title(param_name)
#     plt.xlabel("starting value")
#     plt.ylabel("prob density")
# plt.legend()
# plt.tight_layout()






# ## functions 
# def exponential_decay(t,N0,lbd):
#     err = N0*np.exp(-lbd*t)
#     return err

# def expo_halflife(lbd):
#     return np.exp(2)/lbd 

# def error_expdecay_loglikelihood(t,errs,N0,lbd,sigma):
#     # for truncated norm, the a,b parameter should be specified as follows:
#     # a_trunc and b_trunc are the abscissae at which we wish to truncate the distribution
#     # a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
#     a_trunc,b_trunc = (0,1) # truncate error at 0,1
#     err_mus = exponential_decay(t,N0,lbd)
#     alist, blist = (a_trunc - err_mus) / sigma, (b_trunc - err_mus) / sigma
#     ll = [truncnorm.logpdf(er,a,b,ermu,sigma) for er,a,b,ermu in zip(errs,alist,blist,err_mus)]
#     return np.sum(ll)

# param_bound = dict(zip(["N0","decayrate"],
#                        [(1e-4,1),(1e-4,4*np.log(5))] ))#bounds of param "N0","lbd","sigma", max bound of lambda is set by assuming decreasing from maxN0 to minN0 in two-steps


# def generate_errorseq_fromparam(t,N0,lbd,sigma):
#     err_mus = exponential_decay(t,N0,lbd)
#     a_trunc,b_trunc = (0,1)
#     alist, blist = (a_trunc - err_mus) / sigma, (b_trunc - err_mus) / sigma
#     errseq = [truncnorm.rvs(a,b,ermu,sigma,size=1) for a,b,ermu in zip(alist,blist,err_mus)]
#     return np.hstack(errseq)

# def grab_init_guess_sigmoid(t,err,param_bounds=[]):
#     sint = 1e-8# this is to avoid log zero
#     lny = np.log(err+sint)
#     linres = scipy.stats.linregress(x=t,y=lny)
#     lbd = -linres.slope
#     N0 = np.exp(linres.intercept)

#     lny_pred = linres.slope*t + linres.intercept - sint
#     err_pred = np.exp(lny_pred)
#     sigma = np.std(err_pred-err)
#     p0s = [N0,lbd,sigma]
    
#     param_bounds = [(-np.inf,np.inf) for _ in range(5)] if len(param_bounds)==0 else param_bounds
#     assert len(param_bounds) == 5
#     bounded_p0s = []
#     for p0,(pmin,pmax) in zip(p0s,param_bounds):
#         if p0<pmin or p0>pmax:
#             p0 = np.random.rand(1)[0]*(pmax-pmin)+pmin
#         #p0 = min([p0,pmax])
#         #p0 = max([p0,pmin])
#         bounded_p0s.append(p0)
    
#     return bounded_p0s


# res_df_list = []
# for subid, subdf in error_data.groupby("subid"):
#     subg = "Generalizer" if subid in generalizers else "nonGeneralizer"
#     subc = "First Cohort" if subid in cohort1ids else "Second Cohort"
#     for yvar in ["error_x_rescaled","error_y_rescaled"]:
#         print(subid,yvar)
#         odd_data = subdf[np.mod(subdf["trainingtrialid"],2)==1].copy()
#         even_data = subdf[np.mod(subdf["trainingtrialid"],2)==0].copy()
#         for dsname, ds in zip(["odd","even"],[odd_data,even_data]):
#             xdata = ds["trainingtrialid_scaled"].to_numpy()
#             ydata = ds[yvar].to_numpy()

#             objective_fun = lambda params: np.mean((ydata - exponential_decay(xdata,*params))**2)
#             #objective_fun = lambda params: -error_expdecay_loglikelihood(xdata,ydata,*params)
#             p0 = grab_init_guess_sigmoid(xdata,ydata) # this is an mandatory initial guess

#             # optimal_solution, optimal_fval = multi_start_optimisation(objective_func=objective_fun,
#             #                   bounds=list(param_bound.values())[:2],
#             #                   nstart=10000,
#             #                   n_jobs=10)
#             optimal_solution, optimal_fval =wise_start_optimisation(objective_func=objective_fun,x0=p0,bounds= list(param_bound.values()))

#             pred_err = exponential_decay(xdata,*optimal_solution)#generate_errorseq_fromparam(xdata,*optimal_solution)
#             r2_predtrue = scipy.stats.linregress(x=ydata,y=pred_err).rvalue**2

#             resdf = pd.DataFrame()
#             resdf["param"] = list(param_bound.keys())
#             resdf["estimates"] = optimal_solution
#             resdf = resdf.assign(subid=subid,subgroup=subg,subcohort=subc,datasplit=dsname,yvar=yvar,r2=r2_predtrue)

#             res_df_list.append(resdf)

# res_df_all = pd.concat(res_df_list,axis=0).reset_index(drop=True)
# res_df_all.to_csv(os.path.join(studydata_dir,"pretrainerrorfit.csv"))



# def error_loglikelihood(t,errs,s,b,sigma):
#     # for truncated norm, the a,b parameter should be specified as follows:
#     # a_trunc and b_trunc are the abscissae at which we wish to truncate the distribution
#     # a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
#     a_trunc,b_trunc = (0,1) # truncate error at 0,1
#     err_mus = sigmoid(t,s,b)
#     alist, blist = (a_trunc - err_mus) / sigma, (b_trunc - err_mus) / sigma
#     ll = [truncnorm.logpdf(er,a,b,ermu,sigma) for er,a,b,ermu in zip(errs,alist,blist,err_mus)]
#     return np.sum(ll)

# def generate_errorseq_fromparam(t,s,b,sigma):
#     err_mus = sigmoid(t,s,b)
#     a_trunc,b_trunc = (0,1)
#     alist, blist = (a_trunc - err_mus) / sigma, (b_trunc - err_mus) / sigma
#     errseq = [truncnorm.rvs(a,b,ermu,sigma,size=1) for a,b,ermu in zip(alist,blist,err_mus)]
#     return np.hstack(errseq)