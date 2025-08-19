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
# def multi_start_presetgrid_optimisation(xdata,ydata,y_sliding_se,bounds,preset_x0_grids=list,
#                                  n_jobs=1):
#     rand_starts = preset_x0_grids
#     optim_func = lambda x0: scipy.optimize.curve_fit(f=sigmoid,xdata=xdata,ydata=ydata,
#                             p0=x0,
#                             #sigma=y_sliding_se,
#                             bounds = ([bnd[0] for bnd in bounds.values()],[bnd[1] for bnd in bounds.values()]),
#                             loss="soft_l1",full_output=True,max_nfev=5000)
#     # multi-start grid search optimization
#     with Parallel(n_jobs=n_jobs) as parallel:
#         oplist = parallel(delayed(optim_func)(x0) for x0 in rand_starts)
#     objfun_vals,solutions = zip(*[(np.sum(res[2]["fvec"]),res[0]) for res in oplist])
#     optimal_solution      = solutions[np.argmin(objfun_vals)]
#     optimal_fval          = objfun_vals[np.argmin(objfun_vals)]
#     return optimal_solution, optimal_fval

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


for runon_stage in ["train"]: # "train",
    if runon_stage == "test":
        param_bound = dict(zip(["u","s","b"],
                        [(1e-5,1),(-4,-1e-5),(1,6*16)] ))#bounds of param capacity, slope,inflection point
    elif runon_stage == "train":
        param_bound = dict(zip(["u","s","b"],
                        [(1e-5,1),(-4,-1e-5),(1,12*9)] ))#bounds of param capacity, slope,inflection point
    Ngrids = dict(zip(["u","s","b"], [10,40,int(param_bound["b"][1]/2)]))
    #Ngrids = dict(zip(["u","s","b"], [5,30,30] ))

    error_data = pd.read_csv(os.path.join(studydata_dir,f"trialwisepretrain{runon_stage}data.csv")).drop(columns=["X","Unnamed..0"])
    res_df_list = []
    for subid, subdf in error_data.groupby("subid"):
        subg = "Generalizer" if subid in generalizers else "nonGeneralizer"
        subc = "First Cohort" if subid in cohort1ids else "Second Cohort"
        for yvar in ["error_x_rescaled","error_y_rescaled"]:
            print(subc, subg, subid,yvar)
            xvar = "trainingtrialid" if runon_stage =="train" else "testtrialid"
            odd_data = subdf[np.mod(subdf[xvar],2)==1].copy()
            even_data = subdf[np.mod(subdf[xvar],2)==0].copy()
            for dsname, ds in zip(["odd","even"],[odd_data,even_data]):
                xdata = ds[xvar].to_numpy()
                ydata = ds[yvar].to_numpy()

                y_sliding_window = [ydata[max(j-4,0):min(j+4,len(ydata)-1)] for j in range(len(ydata))]
                y_sliding_se = [ys.std()/np.sqrt(ys.size) for ys in y_sliding_window]
                
                # guess starting value for estimating the warm start for grid search            
                p0 = grab_init_guess_sigmoid(xdata,ydata)
                #print(f"{dsname} First Guess: {['%.3f'%x for x in p0]}")

                # get the warm start for grid search using nls
                #try:
                    # lsguess_objfun = lambda params: (ydata - sigmoid(xdata,*params))
                    # lsguess = scipy.optimize.least_squares(fun=lsguess_objfun,
                    #                             x0=get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="random"),
                    #                             bounds = ([bnd[0] for bnd in param_bound.values()],[bnd[1] for bnd in param_bound.values()]),
                    #                             loss="soft_l1")
                    # warmstart = lsguess.x
                    # lsguess = scipy.optimize.curve_fit(f=sigmoid,xdata=xdata,ydata=ydata,
                    #                          p0=get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="random"),
                    #                          sigma=y_sliding_se,
                    #                          bounds = ([bnd[0] for bnd in param_bound.values()],[bnd[1] for bnd in param_bound.values()]),
                    #                          loss="soft_l1")
                    # warmstart = lsguess[0]
                    
                    # soft_l1 = lambda z: 2*((1+z)**0.5-1)
                    # cauchy = lambda z: np.log(1+z)
                    # arctan = lambda z: arctan(1+z)
                    # lsguess_objfun = lambda params: 0.5*np.sum(soft_l1(((ydata - sigmoid(xdata,*params)))**2))
                    # lsguess = scipy.optimize.minimize(lsguess_objfun,
                    #                                   x0=get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="random"),
                    #                                   bounds=list(param_bound.values()))
                    #warmstart = lsguess.x
                #except:
                #    warmstart = get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="random") 
                warmstart = get_bounded_vals(p0,list(param_bound.values()),flag_hitbound="random") 
                warmstartsigs = [(bnd[1]-bnd[0])/6 for bnd in param_bound.values()]
                
                #print(f"{dsname} Second Guess: {['%.3f'%x for x in warmstart]}")
                
                # grid-search with warm start      
                #grid_edges = [sample_paramgrid(mu, sig, bnd, Ng) for mu,sig, bnd, Ng in zip(warmstart,warmstartsigs,param_bound.values(),Ngrids.values())]
                grid_edges = [sample_paramgrid_from_uniform(bnd, Ng) for mu,sig, bnd, Ng in zip(warmstart,warmstartsigs,param_bound.values(),Ngrids.values())]
                #grid_edges = []
                #for mu,sig, pname,bnd, Ng in zip(warmstart,warmstartsigs,param_bound.keys(),param_bound.values(),Ngrids.values()):
                #    if pname != "s":
                #        grid_edges.append(sample_paramgrid(mu, sig, bnd, Ng))
                #    else:
                #        grid_edges.append(sample_paramgrid_from_uniform(bnd, Ng))

                grid_lbs = [g[:-1] for g in grid_edges]
                grid_ubs = [g[1:] for g in grid_edges]
                x0s = [[np.random.uniform(low=glb,high=gub) for glb,gub in zip(pglb,pgub)] for pglb,pgub in zip(grid_lbs,grid_ubs)]
                #for j,(x0, (pname,pbnd)) in enumerate(zip(x0s,param_bound.items())):
                #    plt.subplot(1,len(param_bound),j+1)
                #    plt.scatter(np.arange(len(x0)),x0)
                #    plt.axhline(pbnd[0])
                #    plt.axhline(pbnd[1])
                #    plt.title(pname)
                #plt.tight_layout()

                x0_grids = list(itertools.product(*x0s))
                ###### multi_start_presetgrid_optimisation calls `minimize` in scipy, so it requires an objective funtion that returns a scalar
                soft_l1 = lambda z: 2*((1+z)**0.5-1)
                cauchy = lambda z: np.log(1+z)
                arctan = lambda z: arctan(1+z)
                gsmin_objfun_soft_l1 = lambda params: np.sum(soft_l1(((ydata - sigmoid(xdata,*params)))**2))
                gsmin_objfun_mse = lambda params: np.sum(((ydata - sigmoid(xdata,*params)))**2)
                
                optimal_solution, optimal_fval = multi_start_presetgrid_optimisation(objective_func=gsmin_objfun_soft_l1,
                                bounds=list(param_bound.values()),
                                preset_x0_grids=x0_grids,
                                n_jobs=16)
                #optimal_solution, optimal_fval = multi_start_presetgrid_optimisation(xdata,ydata,y_sliding_se,param_bound,preset_x0_grids=x0_grids,
                #                 n_jobs=16)

                pred_err = sigmoid(xdata,*optimal_solution)
                r2_predtrue = scipy.stats.linregress(x=ydata,y=pred_err).rvalue**2
                r2_ceil = scipy.stats.linregress(odd_data[yvar],even_data[yvar]).rvalue**2
                print(f"{dsname} Estimation found: {['%.3f'%x for x in optimal_solution]}, R2={'%.3f'%r2_predtrue}, R2ceil={'%.3f'%r2_ceil}, R2/ceil ={'%.3f'%(r2_predtrue/r2_ceil)} ")

                resdf = pd.DataFrame()
                resdf["param"] = list(param_bound.keys())
                resdf["estimates"] = list(optimal_solution)
                resdf = resdf.assign(subid=subid,subgroup=subg,subcohort=subc,datasplit=dsname,yvar=yvar,r2=r2_predtrue,r2ceil = r2_ceil)

                res_df_list.append(resdf)

    res_df_all = pd.concat(res_df_list,axis=0).reset_index(drop=True)
    res_df_all.to_csv(os.path.join(studydata_dir,f"pretrain{runon_stage}errorfit_unigs_softl1loss.csv"))


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