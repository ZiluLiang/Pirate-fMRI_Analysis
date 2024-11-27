import numpy
import scipy
from scipy.optimize import minimize,brute
from joblib import Parallel, delayed
from typing import Union
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import pandas
from copy import deepcopy
import itertools
from sklearn.linear_model import LinearRegression
from functools import partial

def _compose_pattern_from_reference_single_source(source_feature:Union[numpy.ndarray,list],
                                                  R:numpy.ndarray,
                                                  M:numpy.ndarray,
                                                  source_controlfeature:Union[numpy.ndarray,list]=[],
                                                  M0:numpy.ndarray=[])->numpy.ndarray:
    """compose pattern for a single source from the reference matrix by matching reference features with the source's features. \n
    Let ``R`` be the reference pattern matrix that contains representation patterns of different feature combinations in the same or a different task.
    Let ``f(x,y)`` be a row vector in ``R`` that correspond to the representation pattern of feature combination represented by vector``(x,y)`` in the voxel space. \n
    We can compose the representation pattern of new feature combination ``(a,b)`` from ``R`` by:\n
      1) selecting rows in ``R`` that contains feature ``a``, same for ``b`` \n
      2) averaging the representation patterns of ``a``, same for ``b`` \n
      3) concatenating the average representation patterns of ``a`` and ``b`` \n
    In other words:    ``g(a,b) = [f(x,y|x=a),f(x,y|y=b)]``   \n
    where ``(a,b)`` is the source feature vector. the reference features matrix ``M`` stores all the information about the features of each rows of ``R``. \n

    In addition, pattern selection further constrained by control features:
    Given ``source_controlfeature``=(c,...), the search of patterns in ``R`` is further limited to make sure only rows where M0 == c is included in the construction of compositional patterns
    
    Parameters
    ----------
    source_feature : Union[numpy.ndarray,list]
        features of the source stimulus that are used to find the corresponding rows in the reference matrix to compose pattern. size =`n`
    R : numpy.ndarray
        the reference pattern matrix. a 2D numpy array that has ``m`` rows. Each row is the representation pattern of a reference stimuli with features specified in the corresponding row of ``M``
    M : numpy.ndarray
        the reference features matrix, a 2D numpy array of shape ``(m,n)`` Each row is the features of a reference stimuli 
    source_controlfeature : Union[numpy.ndarray,list], optional
        control features, by default []
    M0 : numpy.ndarray, optional
        the reference control features matrix specifying control features of the reference samples, by default []

    Returns
    -------
    numpy.ndarray
        composed feature matrix
    """

    source_feature = numpy.array(source_feature).flatten()
    source_controlfeature = numpy.array(source_controlfeature).flatten()
    R = numpy.array(R)
    M = numpy.array(M)
    M0 = numpy.array(M0)

    assert source_feature.size == M.shape[1]
    assert R.shape[0] == M.shape[0]
    if source_controlfeature.size>0:
        assert R.shape[0] == M0.shape[0]
        assert source_controlfeature.size == M0.shape[1]
        #the rows in the reference stimuli where the control feature matches the sample
        ctrl_f_filter = [numpy.array_equal(ref_ctrlfs, source_controlfeature) for ref_ctrlfs in M0]
    else:
         ctrl_f_filter = [True] * R.shape[0]
    
    
    f_filters = [
        [all([x==f,cf]) for x,cf in zip(M[:,j],ctrl_f_filter)] for j,f in enumerate(source_feature)
    ]

    vecs = [numpy.mean(R[f_filter,:],axis=0).flatten() for f_filter in f_filters]
    return numpy.concatenate(vecs)

def compose_pattern_from_reference(source_features:numpy.ndarray,
                                   reference_pattern:numpy.ndarray,reference_features:numpy.ndarray,
                                   source_controlfeatures:numpy.ndarray=[],reference_controlfeatures:numpy.ndarray=[]):
    """	compose pattern from the reference matrix by matching reference features with the sources' features. \n
    Parameters
    ----------
    source_features : Union[numpy.ndarray,list]
        features of k source stimuli that are used to find the corresponding rows in the reference matrix to compose pattern. a 2D numpy array of shape `(k,n)`
    reference_pattern : numpy.ndarray
        the reference pattern matrix. a 2D numpy array that has ``m`` rows. Each row is the representation pattern of a reference stimuli with features specified in the corresponding row of ``M``
    reference_features : numpy.ndarray
        the reference features matrix, a 2D numpy array of shape ``(m,n)`` Each row is the features of a reference stimuli 
    source_controlfeatures : Union[numpy.ndarray,list], optional
        control features,a 2D numpy array of shape `(k,p)`, by default []
    reference_controlfeatures : numpy.ndarray, optional
        the reference control features matrix specifying control features of the reference samples,a 2D numpy array of shape `(m,p)`, by default []

    Returns
    -------
    numpy.ndarray
        composed feature matrix

    """
    
    source_features        = numpy.atleast_2d(source_features)
    source_controlfeatures = numpy.atleast_2d(source_controlfeatures)
    if source_controlfeatures.size>0:
        assert source_controlfeatures.shape[0] == source_features.shape[0]
        return numpy.array([_compose_pattern_from_reference_single_source(sf,reference_pattern,reference_features,scf,reference_controlfeatures) for sf,scf in zip(source_features,source_controlfeatures)])
    else:
        return numpy.array([_compose_pattern_from_reference_single_source(sf,reference_pattern,reference_features) for sf in source_features])

from scipy.stats import truncnorm,uniform
def univariate_truncated_gaussian(x:float,mu:float,sigma:float,lb:float,ub:float)->float:
    """_summary_

    Parameters
    ----------
    x : float
        observed data
    mu : float
        mean of the distribution
    sigma : float
        standard deviation of the distribution, variance = sigma**2
    lb : float
        lower bound of the distribution
    ub : float
        upper bound of the distribution

    Returns
    -------
    float
        pdf of univariate truncated gaussian distribution evaluated at x
    """
    a = (lb-mu)/sigma
    b = (ub - mu) / sigma
    rv = truncnorm(a, b, loc=mu, scale=sigma)
    return rv.pdf(x)

def univariate_uniform(x:float,lb:float,ub:float)->float:
    """_summary_

    Parameters
    ----------
    x : float
        observed data
    lb : float
        lower bound
    ub : float
        upper bound

    Returns
    -------
    float
        pdf of univariate uniform distribution evaluated at x
    """
    rv = uniform(lb, ub-lb)
    return rv.pdf(x)

def disk_uniform(area)->float:
    """probability of sampling uniformly on a disk

    Parameters
    ----------
    area : float
        area of the disk

    Returns
    -------
    float
        probability
    """
    return 1/area

##### different response models    
def p_oneD(alpha,mu,sigma,resp,resp_other,arena_r):
    hr_ = numpy.sqrt(arena_r**2-resp_other**2)    
    if (resp**2+resp_other**2)<=(arena_r**2):
        return univariate_truncated_gaussian(resp, mu,sigma,-hr_, hr_)*(1-alpha) + alpha*univariate_uniform(resp,-hr_,hr_)
    else:
        return numpy.nan

def get_likelihood_function(params:list,param_names:list,xystrategies:list,fixedparams={})->float:
    param_dict  = deepcopy(fixedparams)
    input_param_dict = dict(zip(param_names,params))
    param_dict.update(input_param_dict)
    
    def calculate_bivariate_likelihood(resp_x,resp_y,putative_x,putative_y,arena_r,xstrategy,ystrategy,param_dict):        
        if xstrategy == "putative":
            px = p_oneD(param_dict["lapse_rate"], param_dict["betax"]*putative_x, param_dict["sigma_x"],
                        resp=resp_x, resp_other=resp_y, arena_r=arena_r)
        elif xstrategy == "bias":
            px = p_oneD(param_dict["lapse_rate"], param_dict["biasx"], param_dict["sigma_biasx"],
                        resp=resp_x, resp_other=resp_y, arena_r=arena_r)
        elif xstrategy == "random":
            px = univariate_uniform(resp_x,-numpy.sqrt(arena_r**2-resp_y**2), numpy.sqrt(arena_r**2-resp_y**2))
        
        if ystrategy == "putative":
            py = p_oneD(param_dict["lapse_rate"],param_dict["betay"]*putative_y,param_dict["sigma_y"],
                        resp=resp_y, resp_other=resp_x, arena_r=arena_r)
        elif ystrategy == "bias":
            py = p_oneD(param_dict["lapse_rate"],param_dict["betay"], param_dict["sigma_biasy"],
                        resp=resp_y, resp_other=resp_x, arena_r=arena_r)
        elif ystrategy == "random":
            py = univariate_uniform(resp_y,-numpy.sqrt(arena_r**2-resp_x**2), numpy.sqrt(arena_r**2-resp_x**2) )
        return px*py*(1-param_dict["lapse_rate_2D"]) + disk_uniform(numpy.pi*(arena_r**2))*param_dict["lapse_rate_2D"]


    fs = []

    for xystrategy in xystrategies:
        [xstrategy,ystrategy] = xystrategy.split("-")
        assert xstrategy in ["putative","bias","random"]
        assert ystrategy in ["putative","bias","random"]

        if xystrategy != "random-random":
            curr_bivariate_likelihood_func = partial(calculate_bivariate_likelihood,
                                                    xstrategy=xstrategy,ystrategy=ystrategy,param_dict=param_dict)
        else:
            curr_bivariate_likelihood_func = lambda rx,ry,px,py,arena_r: disk_uniform(numpy.pi*(arena_r**2))

        fs.append(curr_bivariate_likelihood_func)
    likelihood_func = lambda *data: numpy.mean([f(*data) for f in fs])
    return likelihood_func

def compute_llr_across_trials(params:list,param_names:list,xystrategies:list,data:numpy.ndarray,arena_r:float,fixedparams={})->float:
    likelihood_func = get_likelihood_function(params,param_names,xystrategies,fixedparams)
    # each column of data is respxs,respys,putativexs,putativeys
    likelihoods_ = [likelihood_func(rx,ry,px,py,arena_r) for rx,ry,px,py in data]
    # compute the negative loglikelihoods
    # when p is zero, log will return inf, to deal with that, we ask it to return 230 so that it doesn't return inf
    # the value comes from take the negative loglikelihood when p is super super small, here when p=1e-200
    # `-numpy.log(1e-200) = 230`:
    negative_loglikelihoods = [-numpy.log(p) if p>0 else 230 for p in likelihoods_] 
    return numpy.sum(negative_loglikelihoods)

def wise_start_optimisation(objective_func,bounds,args,x0,minimize_args={}):
    print("starting val: ",x0)
    def_minimize_args = {}
    def_minimize_args.update(minimize_args)
    res = minimize(objective_func,x0,args=args, bounds=bounds,**def_minimize_args)
    return res.x, res.fun

def multi_start_presetgrid_optimisation(objective_func,bounds,args,preset_x0_grids=list,minimize_args={},
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

def multi_start_optimisation(objective_func:callable,bounds:tuple,args=(),Ns:int=10,nstart:int=None,n_jobs=1,
                             minimize_args = {}):
    """start over multiple grids within the parameter range to find the optimal parameters

    Parameters
    ----------
    objective_func : callable
        The objective function to be minimized
    bounds : tuple
        Each component of the ``bounds`` tuple must be a range tuple of the form (low, high). 
        This function uses these to create the grid of points on which the objective function will be computed. 
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the function, by default ()
    Ns : int, optional
        Number of grid points along each parameter axes.
        If not otherwise specified, will use `nstart` to determine the number of grids.
        If neither `Ns` or `nstart` is specified, will use a default corresponding to `nstart` roughly equals to 5000
    nstart : int, optional
        Number of starting points.
        If not specified, will equal to ``Ns ** len(bounds)``.
        If neither `Ns` or `nstart` is specified, will use a default value of 5000

    n_jobs : int, optional
        number of parallel jobs, by default 1

    Returns
    -------
    tuple
        a tuple of (optimal_solution, optimal_fval)
    """
    # random generator
    rng = numpy.random.default_rng()
    # set up parameter grids
    Np = len(bounds)
    if nstart is not None:
        Ns = int(numpy.floor(nstart**(1/Np)))
    else:
        nstart = Ns**Np
        if nstart > 10000:
            Ns = int(numpy.floor(10000**(1/Np)))
            nstart = Ns**Np
        print(f"{Ns} grid points along each parameter axes, {nstart} start points in total")
    bin_widths = [(cbnd[1]-cbnd[0])/Ns for cbnd in bounds]
    x0s =[[rng.uniform(low=cbnd[0]+bw*k, high=cbnd[0]+bw*(k+1))  for k in range(Ns)] for cbnd,bw in zip(bounds,bin_widths)]
    x0_grids = list(itertools.product(*x0s))
    x0_grid_idx = rng.permutation(numpy.arange(len(x0_grids)))
    rand_starts = [x0_grids[k] for k in x0_grid_idx]

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

def BIC(NLL:float,N:int,k:int)->float:
    """
    NLL: negative log-likelihood of the model
    N: number of examples in the training dataset
    k: number of parameters in the model
    """
    return 2 * NLL + numpy.log(N) * k

def AIC(NLL:float,N:int,k:int)->float:
    """
    NLL: negative log-likelihood of the model
    N: number of examples in the training dataset
    k: number of parameters in the model
    """
    return 2 * NLL + 2 * k

################### sample from response models
def sample_from_mixture_distribution(submodels:list,submargs:list,weights:list,samplesize:int=1,random_state=None):
    weights = numpy.array(weights)/numpy.array(weights).sum()
    rs = check_random_state(random_state)
    submodel_choices = numpy.random.choice(len(submodels), size=samplesize, p = weights)
    submodel_samples = [submodel(samplesize=samplesize,random_state=rs,**submarg) for submodel,submarg in zip(submodels,submargs)]
    rvs = [submodel_samples[kmodel][jsample] for jsample,kmodel in enumerate(submodel_choices)]
    return rvs


def bivariate_sample_from_disk(radius,samplesize,random_state=None):
    rs = check_random_state(random_state)
    r = rs.uniform(low=0, high=1, size=samplesize)
    theta = rs.uniform(low=0, high=2*numpy.pi, size=samplesize)  # angle
    x = numpy.sqrt(r)*radius * numpy.cos(theta)
    y = numpy.sqrt(r)*radius * numpy.sin(theta)
    return x,y


from scipy._lib._util import check_random_state
      

def get_bound(a,r):
    return numpy.sqrt(r**2-a**2)    

def get_univarite_response_sampling_function(strategy,resp_other,r,alpha,putative_resp,bias,sigma,sigma_bias):
    mu = putative_resp
    DIST_CATEGORIES = {
        "putative": lambda samplesize,random_state: truncnorm.rvs((-get_bound(resp_other,r)-mu)/sigma, (get_bound(resp_other,r)-mu)/sigma, loc=mu, scale=sigma,size=samplesize,random_state=random_state),
        "bias":     lambda samplesize,random_state: truncnorm.rvs((-get_bound(resp_other,r)-bias)/sigma_bias, (get_bound(resp_other,r)-bias)/sigma_bias, loc=bias, scale=sigma_bias,size=samplesize,random_state=random_state),
        "random":   lambda samplesize,random_state: uniform.rvs(-get_bound(resp_other,r), 2*get_bound(resp_other,r),size=samplesize,random_state=random_state)
    }
    if strategy == "putative":
        return lambda samplesize,random_state: sample_from_mixture_distribution(
                                    submodels = [DIST_CATEGORIES["putative"], DIST_CATEGORIES["random"]],
                                    submargs=[{},{}],
                                    weights=[1-alpha,alpha],samplesize=samplesize,random_state=random_state)
    if strategy == "bias":
        return lambda samplesize,random_state: sample_from_mixture_distribution(
                                    submodels = [DIST_CATEGORIES["bias"], DIST_CATEGORIES["random"]],
                                    submargs=[{},{}],
                                    weights=[1-alpha,alpha],samplesize=samplesize,random_state=random_state)
    elif strategy == "random":
        return DIST_CATEGORIES["random"]

def sample_response(param_dict:dict,xystrategies:list,data:numpy.ndarray,arena_r:float,sample_size,seed=None):
    random_state = check_random_state(seed)
    nD = data.shape[0]
    # unpack parameters
    if numpy.any([xy!="random-random" for xy in xystrategies]):
        alpha = param_dict["lapse_rate"]
        betax, betay = param_dict["betax"], param_dict["betay"]
        biasx, biasy = param_dict["biasx"], param_dict["biasy"]
        sx,  sy  = param_dict["sigma_x"],param_dict["sigma_y"]
        sbx, sby = param_dict["sigma_biasx"],param_dict["sigma_biasy"]

    rxs,rys,pxs,pys= [data[:,k] for k in range(4)]

    def curr_bivariate_strategy(kwargsx,kwargsy,samplesize,random_state,xstrategy,ystrategy):
        funcx = partial(get_univarite_response_sampling_function,
                        strategy=xstrategy, r=arena_r, alpha=alpha, bias=biasx, sigma=sx, sigma_bias=sbx)
        # still need to pass{"resp_other":ry,"putative_resp":px}
        funcy = partial(get_univarite_response_sampling_function,
                        strategy=ystrategy, r=arena_r, alpha=alpha, bias=biasy, sigma=sy, sigma_bias=sby)
        return numpy.vstack([funcx(**kwargsx)(samplesize,random_state),funcy(**kwargsy)(samplesize,random_state)]).T
            
    strategy_sampling_funcs = []
    for xystrategy in xystrategies:
        if xystrategy == "random-random":
            strategy_sampling_funcs.append(lambda samplesize,random_state,**kwargs: numpy.vstack(bivariate_sample_from_disk(arena_r,samplesize,random_state)).T)
        else:
            [xstrategy,ystrategy] = xystrategy.split("-")
            strategy_sampling_funcs.append(partial(curr_bivariate_strategy,xstrategy=xstrategy,ystrategy=ystrategy))

    # add the noncompo random model
    if numpy.any([xy!="random-random" for xy in xystrategies]):
        strategy_sampling_funcs.append(lambda samplesize,random_state,**kwargs: numpy.vstack(bivariate_sample_from_disk(arena_r,samplesize,random_state)).T)

    sampled_resp = numpy.full((sample_size,2,nD),fill_value=numpy.nan)
    for j, (px,py,rx,ry) in enumerate(zip(pxs,pys,rxs,rys)):
        sargs = {"kwargsx":{"resp_other":ry,"putative_resp":px},
                 "kwargsy":{"resp_other":rx,"putative_resp":py}}
        if numpy.any([xy!="random-random" for xy in xystrategies]):
            sampled_resp[:,:,j] = sample_from_mixture_distribution(
                                submodels=strategy_sampling_funcs,
                                submargs=[sargs for _ in range(len(xystrategies)+1)],
                                weights=[(1-alpha)/len(xystrategies) for _ in range(len(xystrategies))] + [alpha],
                                samplesize=sample_size,random_state=random_state)
        else:
            sampled_resp[:,:,j] = sample_from_mixture_distribution(
                                submodels=strategy_sampling_funcs,
                                submargs=[sargs for _ in range(len(xystrategies))],
                                weights=[1/len(xystrategies) for _ in range(len(xystrategies))],
                                samplesize=sample_size,random_state=random_state)
        
    return sampled_resp
        
def calculate_r_squared_univariate(y,yh): 
    ss_res   = ((y-yh)**2).sum()
    ss_total = ((y-y.mean())**2).sum()
    return 1 - ss_res/ss_total
            
def calculate_r_squared_bivariate(y,yh,ds_axis=0): 
    ss_res   = ((y-yh)**2).sum()
    ss_total = ((y-y.mean(axis=ds_axis))**2).sum()
    return 1 - ss_res/ss_total

def calculate_regression(y,yh):
    slope, _, r, _, _ = scipy.stats.linregress(y.flatten(),yh.flatten())
    return slope


# def trialwise_likelihood_compositional(sigmax:float,sigmay:float,
#                                        resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     """2D model where:
#     x is sampled around the putative x
#     y is sampled around the putative y

#     Parameters
#     ----------
#     sigmax : float
#         standard deviation of gaussian distribution that governs how response on x axis is sampled around the putative x location
#     sigmay : float
#         standard deviation of gaussian distribution that governs how response on y axis is sampled around the putative y location
#     resp_x : float
#         observed response location on x axis
#     resp_y : float
#         observed response location on y axis
#     putative_x : float
#         putative response location on x axis
#     putative_y : float
#         putative response location on x axis
#     arena_r : float
#         radius of the arena. It is used to determine the valid x/y range given y/x.

#     Returns
#     -------
#     float
#         pdf of model evaluated at resp_x and resp_y
#     """
#     hr_x = numpy.sqrt(arena_r**2-resp_y**2)
#     hr_y = numpy.sqrt(arena_r**2-resp_x**2)  
#     probx = univariate_truncated_gaussian(resp_x,putative_x,sigmax,-hr_x,hr_x)
#     proby = univariate_truncated_gaussian(resp_y,putative_y,sigmay,-hr_y,hr_y)
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return probx*proby
#     else:
#         return numpy.nan

# def trialwise_likelihood_1Dx_bias(sigmax:float,sigma_biasy:float,biasy:float,
#                                   resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     """1D model where:
#       x is sampled around the putative x
#       y is sampled around the bias y location

#     Parameters
#     ----------
#     sigmax : float
#         standard deviation of gaussian distribution that governs how response on x axis is sampled around the putative x location
#     sigma_biasy : float
#         standard deviation of gaussian distribution that governs how response on y axis is sampled around the bias y location
#     biasy : float
#         mean of gaussian distribution that governs how response on y axis is sampled around the bias y location
#     resp_x : float
#         observed response location on x axis
#     resp_y : float
#         observed response location on y axis
#     putative_x : float
#         putative response location on x axis
#     putative_y : float
#         putative response location on x axis
#     arena_r : float
#         radius of the arena. It is used to determine the valid x/y range given y/x.

#     Returns
#     -------
#     float
#         pdf of model evaluated at resp_x and resp_y
#     """

#     hr_x = numpy.sqrt(arena_r**2-resp_y**2)
#     hr_y = numpy.sqrt(arena_r**2-resp_x**2)  
#     # x is centered around putative x
#     probx = univariate_truncated_gaussian(resp_x,putative_x,sigmax,-hr_x,hr_x)
#     # y is centered around bias location
#     randy = univariate_truncated_gaussian(resp_y,biasy,sigma_biasy,-hr_y,hr_y)
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return probx*randy
#     else:
#         return numpy.nan

# def trialwise_likelihood_1Dx_random(sigmax:float,
#                                     resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     """1D model where:
#       x is sampled around the putative x
#       y is sampled randomly within valid range

#     Parameters
#     ----------
#     sigmax : float
#         standard deviation of gaussian distribution that governs how response on x axis is sampled around the putative x location
#     resp_x : float
#         observed response location on x axis
#     resp_y : float
#         observed response location on y axis
#     putative_x : float
#         putative response location on x axis
#     putative_y : float
#         putative response location on x axis
#     arena_r : float
#         radius of the arena. It is used to determine the valid x/y range given y/x.

#     Returns
#     -------
#     float
#         pdf of model evaluated at resp_x and resp_y
#     """
#     hr_x = numpy.sqrt(arena_r**2-resp_y**2)
#     hr_y = numpy.sqrt(arena_r**2-resp_x**2)  
#     # x is centered around putative x
#     probx = univariate_truncated_gaussian(resp_x,putative_x,sigmax,-hr_x,hr_x)
#     # y is randomly chosen within allowed range
#     randy = univariate_uniform(resp_y,-hr_y,hr_y)
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return probx*randy
#     else:
#         return numpy.nan
    
# def trialwise_likelihood_1Dy_bias(sigma_biasx:float,sigmay:float,biasx:float,
#                                   resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     """1D model where:
#       x is sampled around the bias x
#       y is sampled around the putative y

#     Parameters
#     ----------
#     sigma_biasx : float
#         standard deviation of gaussian distribution that governs how response on x axis is sampled around the bias x location
#     sigmay : float
#         standard deviation of gaussian distribution that governs how response on y axis is sampled around the putative y location
#     biasx : float
#         mean of gaussian distribution that governs how response on x axis is sampled around the bias x location
#     resp_x : float
#         observed response location on x axis
#     resp_y : float
#         observed response location on y axis
#     putative_x : float
#         putative response location on x axis
#     putative_y : float
#         putative response location on x axis
#     arena_r : float
#         radius of the arena. It is used to determine the valid x/y range given y/x.

#     Returns
#     -------
#     float
#         pdf of model evaluated at resp_x and resp_y
#     """
#     hr_x = numpy.sqrt(arena_r**2-resp_y**2)
#     hr_y = numpy.sqrt(arena_r**2-resp_x**2)  
#     # x is centered around bias location
#     randx = univariate_truncated_gaussian(resp_x,biasx,sigma_biasx,-hr_x,hr_x)
#     # y is centered around putative y
#     proby = univariate_truncated_gaussian(resp_y,putative_y,sigmay,-hr_y,hr_y)
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return randx*proby
#     else:
#         return numpy.nan

# def trialwise_likelihood_1Dy_random(sigmay:float,resp_x:float,resp_y:float,
#                                     putative_x:float,putative_y:float,arena_r:float)->float:
#     """1D model where:
#       x is sampled randomly within valid range
#       y is sampled around the putative y

#     Parameters
#     ----------
#     sigmay : float
#         standard deviation of gaussian distribution that governs how response on y axis is sampled around the putative y location
#     resp_x : float
#         observed response location on x axis
#     resp_y : float
#         observed response location on y axis
#     putative_x : float
#         putative response location on x axis
#     putative_y : float
#         putative response location on x axis
#     arena_r : float
#         radius of the arena. It is used to determine the valid x/y range given y/x.

#     Returns
#     -------
#     float
#         pdf of model evaluated at resp_x and resp_y
#     """
#     hr_x = numpy.sqrt(arena_r**2-resp_y**2)
#     hr_y = numpy.sqrt(arena_r**2-resp_x**2)  
#     # x is randomly chosen within allowed range
#     randx = univariate_uniform(resp_x,-hr_x,hr_x)
#     # y is centered around putative y
#     proby = univariate_truncated_gaussian(resp_y,putative_y,sigmay,-hr_y,hr_y)
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return randx*proby
#     else:
#         return numpy.nan

# def trialwise_likelihood_1D_bias(sigmax:float,sigmay:float,sigma_bias:float,biasx:float,biasy:float,
#                                  resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     """combination of the two 1D bias model,

#     Parameters
#     ----------
#     sigmax : float
#         standard deviation of gaussian distribution that governs how response on x axis is sampled around the putative x location
#     sigmay : float
#         standard deviation of gaussian distribution that governs how response on y axis is sampled around the putative y location
#     sigma_biasx : float
#         standard deviation of gaussian distribution that governs how response on x axis is sampled around the bias x location
#     sigma_biasy : float
#         standard deviation of gaussian distribution that governs how response on y axis is sampled around the bias y location
#     biasx : float
#         mean of gaussian distribution that governs how response on x axis is sampled around the bias x location
#     biasy : float
#         mean of gaussian distribution that governs how response on y axis is sampled around the bias y location
#     resp_x : float
#         observed response location on x axis
#     resp_y : float
#         observed response location on y axis
#     putative_x : float
#         putative response location on x axis
#     putative_y : float
#         putative response location on x axis
#     arena_r : float
#         radius of the arena. It is used to determine the valid x/y range given y/x.

#     Returns
#     -------
#     float
#         pdf of model evaluated at resp_x and resp_y
#     """
#     prob_1Dx = trialwise_likelihood_1Dx_bias(sigmax,sigma_bias,biasy,resp_x,resp_y,putative_x,putative_y,arena_r)
#     prob_1Dy = trialwise_likelihood_1Dy_bias(sigma_bias,sigmay,biasx,resp_x,resp_y,putative_x,putative_y,arena_r)
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return prob_1Dx + prob_1Dy
#     else:
#         return numpy.nan

# def trialwise_likelihood_1D_random(sigma:float,
#                                    resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     prob_1Dx = trialwise_likelihood_1Dx_random(sigma,resp_x,resp_y,putative_x,putative_y,arena_r)
#     prob_1Dy = trialwise_likelihood_1Dy_random(sigma,resp_x,resp_y,putative_x,putative_y,arena_r)
    
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return prob_1Dx + prob_1Dy
#     else:
#         return numpy.nan
    

# def trialwise_likelihood_noncomporandom(resp_x:float,resp_y:float,putative_x:float,putative_y:float,arena_r:float)->float:
#     if (resp_x**2+resp_y**2)<=(arena_r**2):
#         return disk_uniform(numpy.pi*(arena_r**2))
#     else:
#         return numpy.nan
# from scipy.stats import rv_continuous
# class MixtureModel(rv_continuous):
#     """ Class for generating mixture model for distribution
#     ref. https://stackoverflow.com/a/72315113
#     """
#     def __init__(self, submodels, *args, weights = None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.submodels = submodels
#         if weights is None:
#             weights = [1 for _ in submodels]
#         if len(weights) != len(submodels):
#             raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
#         self.weights = [w / sum(weights) for w in weights]
        
#     def _pdf(self, x):
#         pdf = self.submodels[0].pdf(x) * self.weights[0]
#         for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
#             pdf += submodel.pdf(x)  * weight
#         return pdf
            
#     def _sf(self, x):
#         sf = self.submodels[0].sf(x) * self.weights[0]
#         for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
#             sf += submodel.sf(x)  * weight
#         return sf

#     def _cdf(self, x):
#         cdf = self.submodels[0].cdf(x) * self.weights[0]
#         for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
#             cdf += submodel.cdf(x)  * weight
#         return cdf     

#     def rvs(self, size):
#         submodel_choices = numpy.random.choice(len(self.submodels), size=size, p = self.weights)
#         submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
#         rvs = numpy.choose(submodel_choices, submodel_samples)
#         return rvs

