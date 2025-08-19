"""
This file runs the cross-validated RSA analysis for the PTA regressors.
The analysis is done in the following steps:
In each permutation:
    1. we randomly picked two axis locations and used them as the pair to calculate PS between x- and y-axis
    2. Data on the selected axis locations were used as fit set (4 stimuli, 2 x and 2 y)
    3. We then calculated the PS for each participant in the fit set
    4. We compared fitPS to the null obtained previously when calculated PS to classify paritcipants
    5. Dependent on participant classification, we then calculated the RSA in the evaluation set (2 stimuli, 1 x and 1 y)
    6. We then extracted the beta weights for different model RDM
    7. We then compute the average beta weights for each participant group for this current permutation
This process is repeated 1000 times to obtain the distribution of the average beta weights for each model RDM in each group
This can be later used to compare against zero, and see if the beta weights are significantly different from zero
"""
import itertools
import numpy as np
import scipy
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import json
from copy import deepcopy
import os
import time
import glob
from joblib import Parallel, delayed, cpu_count, dump,load

from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri
from zpyhelper.MVPA.preprocessors import split_data
from zpyhelper.MVPA.estimators import MultipleRDMRegression

project_path  = r'D:\OneDrive - Nexus365\pirate_ongoing'
import sys
sys.path.append(project_path)
from scripts.Exp1_fmri.multivariate.pirateOMutils import parallel_axes_cosine_sim,generate_filters
from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Load data
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
studydata_dir = os.path.join(project_path,'data','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    cohort1ids = [x for x in pirate_defaults['participants']['cohort1ids'] if x in subid_list]
    cohort2ids = [x for x in pirate_defaults['participants']['cohort2ids'] if x in subid_list]
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")
print(f"N_cohort1 = {len(cohort1ids)}")
print(f"N_cohort2 = {len(cohort2ids)}")

cohort_names_lists = dict(zip(["First Cohort","Second Cohort","Combined Cohort"],[cohort1ids,cohort2ids,subid_list]))

base_rois = ["HPC","vmPFC","V1","testgtlocParietalSup"]
rois =  [f"{x}_bilateral" for x in base_rois]
rois = ["HPC_bilateral","V1_bilateral"]
ROIRSAdir = os.path.join(project_path,'AALandHCPMMP1andFUNCcluster')
roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))

############ select activity pattern of non-center training stimuli ############ 
sub_patterns = {}
sub_stimdfs = {}
for roi in rois:
    sub_patterns[roi], sub_stimdfs[roi] = [],[]
    for subid,subdata in zip(subid_list,roi_data[roi]):
        print(f" Getting data for {subid} in {roi}",end="\r",flush=True)
        preprocedX = deepcopy(subdata["preprocX"])
        stimdf = subdata["stimdf"].copy()
        
        navi_filter = stimdf.taskname.to_numpy() == "navigation"

        #do not average across sessions, so we get a 100*nvoxel matrix
        navi_X = preprocedX[navi_filter,:]
        
        #filter out the df for just one session
        sub_dfall = stimdf[navi_filter].copy().reset_index(drop=True)
        sub_dfall['training_axsetstr'] = ["x" if axs==0 else "y" for axs in sub_dfall['training_axset']]
        # make loc into integer
        sub_dfall["stim_x"] = sub_dfall["stim_x"]*2
        sub_dfall["stim_y"] = sub_dfall["stim_y"]*2
        sub_dfall["stim_xdist"] = sub_dfall["stim_xdist"]*2
        sub_dfall["stim_ydist"] = sub_dfall["stim_ydist"]*2
        sub_dfall["training_axlocTL"] = sub_dfall["training_axlocTL"]*2
        sub_dfall["training_axlocTR"] = sub_dfall["training_axlocTR"]*2

        #generate filters for various stimuli type
        curr_filt = generate_filters(sub_dfall)["training_nocenter"]
        curr_df, curr_X = sub_dfall[curr_filt].copy().reset_index(drop=True), navi_X[curr_filt,:]

        sub_patterns[roi].append(curr_X)
        sub_stimdfs[roi].append(curr_df)


def cal_PS(X,sdf):
    axlocs = np.unique(sdf.training_axlocTL)
    assert all([np.sum(sdf.stim_x == j)==1 for j in axlocs])
    assert all([np.sum(sdf.stim_y == j)==1 for j in axlocs])
    # pick x/y stims and put them into same order
    xstims = np.array([X[sdf.stim_x == j,:][0] for j in axlocs])
    ystims = np.array([X[sdf.stim_y == j,:][0] for j in axlocs])
    return upper_tri(parallel_axes_cosine_sim(xstims,ystims))[0].mean()

def compute_shuffle(sX,stim_loc,locs,randgenseed=1): 
    stimx = stim_loc[:,0]
    stimy = stim_loc[:,1]
    # initialse random generator for reproducibility
    randomgen = np.random.default_rng(randgenseed)
    pX = sX[randomgen.permutation(sX.shape[0]),:]                
    pxstims = np.array([pX[stimx == j,:][0] for j in locs])
    pystims = np.array([pX[stimy == j,:][0] for j in locs])
    perm_cossims = parallel_axes_cosine_sim(pxstims,pystims,return_codingdirs=False)
    return upper_tri(perm_cossims)[0].mean()


def compute_RSA(subPScat,subX,subdf):
    if subPScat==1:
        PTAhigh_prim = "PTA_locNomial_TL" 
        PTAlow_prim = "PTA_locEuc_TL"
    elif subPScat==-1:
        PTAhigh_prim = "PTA_locNomial_TR"
        PTAlow_prim = "PTA_locEuc_TR"
    else:
        PTAhigh_prim = "PTA_locNomial_TL"
        PTAlow_prim = "PTA_locEuc_TL"

    analyses = [
            {"name": "train stim Compete PTAhigh Cartesian",
            "reg_names": ["PTA_ax", f"{PTAhigh_prim}", "gtlocEuclidean"],
            "short_names":  ["PTA_ax","PTA_locNomial","gtlocEuclidean"]
            },
            
            {"name": "train stim betweenxy PTA low",
            "reg_names": [f"betweenxy_{PTAlow_prim}"],
            "short_names": ["PTA_locEuc"]
            },
            {"name": "train stim betweenxy PTA high",
            "reg_names": [f"betweenxy_{PTAhigh_prim}"],
            "short_names": ["PTA_locNomial"]
            },
            {
            "name": "train stim betweenxy Compete PTA high Cartesian",
            "reg_names": [f"betweenxy_{PTAhigh_prim}",f"betweenxy_gtlocEuclidean"],
            "short_names": ["PTA_locNomial","gtlocEuclidean"]
            }
            
        ]
    stim_dict = subdf.copy().reset_index(drop=True).to_dict('list')
    submodelrdm = ModelRDM(
        stimid      = stim_dict["stim_id"],
        stimgtloc   = np.vstack([stim_dict["stim_x"],stim_dict["stim_y"]]).T,
        stimfeature = np.vstack([stim_dict["stim_color"],stim_dict["stim_shape"]]).T,
        stimgroup   = stim_dict["stim_group"],
        sessions    = stim_dict["stim_session"],
        nan_identity = True,
        splitgroup   =  False
    )
    #print([x for x in submodelrdm.models.keys() if "betweenxy" in x])

    res_list = []
    for analysis_dict in analyses:
        reg_names, short_names = analysis_dict["reg_names"], analysis_dict["short_names"]
        reg_vals  = [submodelrdm.models[k] for k in reg_names]
        reg_estimator  = MultipleRDMRegression(subX,
                                            modelrdms=reg_vals,
                                            modelnames=short_names,
                                            rdm_metric="correlation")
        reg_estimator.fit()
        res_df = pd.DataFrame(
            {"modelrdm":   reg_estimator.modelnames,
            "coefficient": reg_estimator.result}
        ).assign(
        fitPScat = subPScat,
        analysis = analysis_dict["name"],
        adjR2 = reg_estimator.adjR2,
        R2    = reg_estimator.R2
        )
        res_list.append(res_df)
    
    res_df = pd.concat(res_list,axis=0).reset_index(drop=True)
    return res_df

############ classify participants ############ 
def classify_participant(subPS,PSnull,ps_crit = 0.005):
    if np.mean(PSnull<subPS)<ps_crit:
        return -1
    elif np.mean(PSnull>subPS)<ps_crit:
        return 1
    else:
        return 0

########### split data into fit and eval #########
def split_fiteval_data(subX,subdf,evalr):#
    sessions= subdf.stim_session
    assert np.array_equal(np.unique(sessions),np.array([0,1,2,3]))
    fitpair = [x for x in np.unique(sessions) if x not in evalr]
    fit_index = np.array([x in fitpair for x in sessions])
    eval_index = np.array([x in evalr for x in sessions])

    # then we averaged across runs 
    fitdf = subdf[fit_index].copy().reset_index(drop=True)
    fitX = np.mean(split_data(subX[fit_index],fitdf.stim_session),axis=0)
    minfits = fitdf.stim_session.min()
    fitdf = fitdf[fitdf.stim_session==minfits].copy().reset_index(drop=True)

    evaldf = subdf[eval_index].copy().reset_index(drop=True)
    evalX = np.mean(split_data(subX[eval_index],evaldf.stim_session),axis=0)
    minevals = evaldf.stim_session.min()
    evaldf = evaldf[evaldf.stim_session==minevals].copy().reset_index(drop=True)


    return fitX, evalX, fitdf, evaldf


# def calculate_group_mean(subXs,subdfs,subid_list,randseed,savedir,returnsubdfs=False, verbose=False):
#     if verbose:
#         print(f"Calculating for {randseed}",end="\r",flush=True)
#     # all possible pairs to choose as the evaluation pair
#     eval_pairs = list(itertools.combinations([0,1,2,3],2))

#     rng = np.random.default_rng(randseed)        
#     subeval_pairs = []
#     for _ in subid_list:            
#         subeval_pairs.append( rng.permutation(eval_pairs)[0] )

#     psnull = []    
#     ps_fits  = []
#     evalXs,evaldfs = [],[]
#     for jsub, (subX,subdf,subid,eval_pair) in enumerate(zip(subXs,subdfs,subid_list,subeval_pairs)):
                
#         subX_fit, subX_eval, subdf_fit, subdf_eval = split_fiteval_data(subX,subdf,eval_pair)
#         evalXs.append(subX_eval)
#         evaldfs.append(subdf_eval)

#         subPS_fit = cal_PS(subX_fit,subdf_fit)
#         ps_fits.append(subPS_fit)
#         permutations = [compute_shuffle(subX_fit,subdf_fit[["stim_x","stim_y"]].to_numpy(),[-2,-1,1,2],p) for p in range(10000)]
#         psnull.append(permutations)
        
#     psnull = np.mean(psnull,axis=0)
        
#     res_dfs = []
#     for jsub, (subPS_fit,subX_eval,subdf_eval,subid) in enumerate(zip(ps_fits,evalXs,evaldfs,subid_list)):
#         subPScat_fit = classify_participant(subPS_fit,psnull,ps_crit = 0.005)       
#         res_dfs.append(compute_RSA(subPScat_fit,subX_eval,subdf_eval).assign(subid=subid,randseed=randseed))

#     res_df = pd.concat(res_dfs,axis=0).reset_index(drop=True)
#     res_sum = res_df.groupby(["fitPScat","analysis","modelrdm"])["coefficient"].mean().reset_index().assign(randseed=randseed)
#     res_df.to_csv(os.path.join(savedir,f"ROI_PTARSA_CVbyrun_{randseed}.csv"))
#     if returnsubdfs:
#         return res_sum, res_df
#     else:
#         return res_sum

def calculate_group_mean_from_saved(subXs,subdfs,sub_cs_obs,sub_cs_perms,subid_list,randseed,savedir,returnsubdfs=False, verbose=False):
    if verbose:
        print(f"Calculating for permutation {randseed}",end="\r",flush=True)

    # all possible pairs to choose as the evaluation pair
    unique_sessions = [0,1,2,3]
    eval_runs = list(itertools.combinations(unique_sessions,1))

    rng = np.random.default_rng(randseed)        
    subeval_runs = []
    for _ in subid_list:            
        subeval_runs.append( rng.permutation(eval_runs)[0] )

    psnull = []    
    ps_fits  = []
    evalXs,evaldfs = [],[]
    for jsub, (subX,subdf,subPSs,subPERMs,subid,evalr) in enumerate(
        zip(subXs,subdfs,sub_cs_obs,sub_cs_perms,subid_list,subeval_runs)):
        fitpair = [x for x in unique_sessions if x not in evalr]

        subX_fit, subX_eval, subdf_fit, subdf_eval = split_fiteval_data(subX,subdf,evalr)
        evalXs.append(subX_eval)
        evaldfs.append(subdf_eval)

        subPS_fit = np.mean([subPSs[r] for r in fitpair])
        ps_fits.append(subPS_fit)
        
        permutations =[np.mean([subPERMs[p][r] for r in fitpair])  for p in range(10000)]
        psnull.append(permutations)
    psnull = np.mean(psnull,axis=0)
        
    res_dfs = []
    for jsub, (subPS_fit,subX_eval,subdf_eval,subid) in enumerate(zip(ps_fits,evalXs,evaldfs,subid_list)):
        if verbose:
            print(f"Calculating for permutation {randseed} sub {subid}",end="\r",flush=True)
        subPScat_fit = classify_participant(subPS_fit,psnull,ps_crit = 0.005)       
        res_dfs.append(compute_RSA(subPScat_fit,subX_eval,subdf_eval).assign(subid=subid,randseed=randseed))

    res_df = pd.concat(res_dfs,axis=0).reset_index(drop=True)
    res_sum = res_df.groupby(["fitPScat","analysis","modelrdm"])["coefficient"].mean().reset_index().assign(randseed=randseed)
    res_df.to_csv(os.path.join(savedir,f"ROI_PTARSA_CVbyrun_{randseed}.csv"))
    if returnsubdfs:
        return res_sum, res_df
    else:
        return res_sum


########## repeat for each participants in different random seed to generate the distribution
n_perm = 10000
save_fn = "ROI_PTARSA_CVbyrun"
resoutputpar = os.path.join(ROIRSAdir,"axisPSsplitrun")
resoutput_dir = os.path.join(resoutputpar,"RSAsplitrunCV")
roi_res = {}
with Parallel(n_jobs=10) as parallel:
    for pgroi in ["HPC_bilateral"]:
        print(f"Calculating for {pgroi}\n")
        rsaroi = pgroi
        loadedres = load(os.path.join(resoutputpar,f"PS_NCtrainingstim_perrun_{pgroi}.pkl"))
        sub_cs_perms, sub_cs_obs = loadedres["permutation"], loadedres["observation"]
        roisavedir = os.path.join(resoutput_dir,pgroi)
        checkdir(roisavedir)
        parallel(delayed(calculate_group_mean_from_saved)(sub_patterns[pgroi],sub_stimdfs[pgroi],sub_cs_obs,sub_cs_perms,subid_list,perm,roisavedir,True,True) for perm in range(n_perm))
        #dump(roi_res[pgroi],os.path.join(resultdir,f"{save_fn}_{pgroi}.pkl"))

#for pgroi in ["HPC_bilateral"]:
#    roi_res[pgroi] = load(os.path.join(resultdir,f"{save_fn}_{pgroi}.pkl"))
#dump(roi_res,os.path.join(resultdir,f"{save_fn}.pkl"))

# ########## plot the results ##########
save_fn = "ROI_PTARSA_CVbyrun"


#roi_res = load(os.path.join(resultdir,f"{save_fn}.pkl"))
groupave_dfs = {}
groupave_dfs["HPC_bilateral"] = []
for pgroi in ["HPC_bilateral"]:
    roisavedir = os.path.join(resoutput_dir,pgroi)
    csv_list=glob.glob(os.path.join(roisavedir,'*.csv'))
    allsubdf = pd.concat([pd.read_csv(x) for x in csv_list]).reset_index(drop=True)
   

    allsubdf["subgroup"] = ["Generalizer" if x in generalizers else "nonGeneralizer" for x in allsubdf.subid]
    groupavedf = allsubdf.groupby(["randseed","subgroup","fitPScat","analysis","modelrdm"])["coefficient"].mean().reset_index()
    groupave_dfs["HPC_bilateral"].append(groupavedf)

groupO_pta = np.concatenate([df[(df.analysis=="train stim betweenxy Compete PTA high Cartesian")&(df.modelrdm=="PTA_locNomial")&(df.fitPScat==0)&(df.subgroup=="Generalizer")].copy()["coefficient"].values for df in groupave_dfs["HPC_bilateral"]])
groupO_euc = np.concatenate([df[(df.analysis=="train stim betweenxy Compete PTA high Cartesian")&(df.modelrdm=="gtlocEuclidean")&(df.fitPScat==0)&(df.subgroup=="Generalizer")].copy()["coefficient"].values for df in groupave_dfs["HPC_bilateral"]])
groupPS_pta = np.concatenate([df[(df.analysis=="train stim betweenxy Compete PTA high Cartesian")&(df.modelrdm=="PTA_locNomial")&(df.fitPScat!=0)&(df.subgroup=="Generalizer")].copy()["coefficient"].values for df in groupave_dfs["HPC_bilateral"]])
groupPS_euc = np.concatenate([df[(df.analysis=="train stim betweenxy Compete PTA high Cartesian")&(df.modelrdm=="gtlocEuclidean")&(df.fitPScat!=0)&(df.subgroup=="Generalizer")].copy()["coefficient"].values for df in groupave_dfs["HPC_bilateral"]])
rsacv_res = pd.concat([
    pd.DataFrame({"res":groupO_pta}).assign(PScat="orthogonal",modelrdm="PTA_locNomial"),
    pd.DataFrame({"res":groupO_euc}).assign(PScat="orthogonal",modelrdm="gtlocEuclidean"),
    pd.DataFrame({"res":groupPS_pta}).assign(PScat="parallel",modelrdm="PTA_locNomial"),
    pd.DataFrame({"res":groupPS_euc}).assign(PScat="parallel",modelrdm="gtlocEuclidean"),
])
rsacv_res

fig,axes = plt.subplots(1,2,figsize=(8,4))
for j, (ax, (gname,gdf)) in enumerate(zip(axes.flatten(), rsacv_res.groupby("modelrdm"))):
    sns.histplot(data=gdf,x="res",hue="PScat",ax=ax,alpha=0.5)
    ax.set_title(gname)
    xaxlim, yaxlim = ax.get_xlim(), ax.get_ylim()
    for k,(mname,mdf) in  enumerate(gdf.groupby("PScat")):
        ax.text(x=xaxlim[0],y=(1-0.1*(k+1))*yaxlim[1], s=f"{mname}: {'%.3f'% np.mean(mdf.res.values<=0)}")
    sns.move_legend(ax,loc="upper right",bbox_to_anchor=(0,1,1,0.1),title="")











# ############ calculate PS ############            
# eval_pairs = list(itertools.combinations([-2,-1,1,2],2))

# sub_cs_obs_eval = {}
# sub_cs_obs_fit   = {}
# subevaldata_Xs, subevaldata_dfs = {}, {}
# #with Parallel(n_jobs=15) as parallel:
# for pgroi in rois:
#     #sub_cs_perms_fit[pgroi], sub_cs_perms_eval[pgroi] = [],[]
#     sub_cs_obs_fit[pgroi], sub_cs_obs_eval[pgroi] = [], []
#     subevaldata_Xs[pgroi], subevaldata_dfs[pgroi] = [], []
#     for jsub, (subpsX,subdf,subid) in enumerate(zip(sub_patterns[pgroi],sub_stimdfs[pgroi],subid_list)):
#         print(f"\n Calculating PS and PS null in fit set for {subid} in {pgroi}",end="\r",flush=True)
    
#         subpsX_ave = np.mean(split_data(subpsX,subdf.stim_session),axis=0)
#         subdf_singlesess = subdf[subdf.stim_session==0].copy().reset_index(drop=True)
#         subgroup = "Generalizer" if subid in generalizers else "nonGeneralizer"
#         subcohort = "first cohort" if subid in cohort1ids else "second cohort"

#         axlocs = subdf_singlesess.training_axlocTL.values
#         eval_pair = np.random.permutation(eval_pairs)[0]

#         fit_pair = [x for x in np.unique(axlocs) if x not in eval_pair]
#         fit_index = np.array([x in fit_pair for x in axlocs])
#         eval_index = np.array([x in eval_pair for x in axlocs])
        
#         fitPSX  = subpsX_ave[fit_index,:]
#         fitPS_df = subdf_singlesess[fit_index].copy().reset_index(drop=True)

#         evalPSX = subpsX_ave[eval_index,:]
#         evalPS_df = subdf_singlesess[eval_index].copy().reset_index(drop=True)
#         subevaldata_Xs[pgroi].append(evalPSX)
#         subevaldata_dfs[pgroi].append(evalPS_df)
        
#         #compute the PS for fit as well as eval set for later comparison:
#         subfitPS = cal_PS(fitPSX,fitPS_df)
#         eval_PS = cal_PS(evalPSX,evalPS_df)
    
        
#         # by specifying p as the random seed, we obtain the the same random shuffling for each ROI in each participants 
#         #fit_permutations = parallel(delayed(compute_shuffle)(fitPSX,fitPS_df[["stim_x","stim_y"]].to_numpy(),fit_pair,p) for p in range(n_perm))
#         #eval_permutations = parallel(delayed(compute_shuffle)(eval_PS,evalPS_df[["stim_x","stim_y"]].to_numpy(),eval_pair,p) for p in range(n_perm))

#         #sub_cs_perms_fit[pgroi].append(fit_permutations)
#         #sub_cs_perms_eval[pgroi].append(eval_permutations)
#         sub_cs_obs_fit[pgroi].append(subfitPS)
#         sub_cs_obs_eval[pgroi].append(eval_PS)  


# dump({"fit_PS":sub_cs_obs_fit,
#       "eval_PS":sub_cs_obs_eval,
#       #"fit_PSperm":sub_cs_perms_fit,
#       #"eval_PSperm":sub_cs_perms_eval
#       },
#       os.path.join(ROIRSAdir,"ROI_PTARSA_CV.pkl"))        



# dump({"fit_PS":sub_cs_obs_fit,
#       "eval_PS":sub_cs_obs_eval,
#       #"fit_PSperm":sub_cs_perms_fit,
#       #"eval_PSperm":sub_cs_perms_eval,
#       "eval_dataXs":subevaldata_Xs,
#       "eval_dataDfs":subevaldata_dfs,
#       "fit_PScat":fit_subrep_category,
#       "eva_PScat":eval_subrep_category},
#       os.path.join(ROIRSAdir,"ROI_PTARSA_CV.pkl"))        

 



# ############ compute RSA ############
# ROIRSA_PTA_cvres = []
# for pgroi in ["HPC_bilateral"]:  
#     for subid, subeval_X, subeval_df, subfitPScat in zip(subid_list,subevaldata_Xs[pgroi],subevaldata_dfs[pgroi],fit_subrep_category[pgroi]):
#         print(f"evaluating RSA in {subid} in {pgroi}",end="\r",flush=True)
        
#         ## specify analysis
#         if subfitPScat==1:
#             PTAhigh_prim = "PTA_locNomial_TL" 
#             PTAlow_prim = "PTA_locEuc_TL"
#         elif subfitPScat==-1:
#             PTAhigh_prim = "PTA_locNomial_TR"
#             PTAlow_prim = "PTA_locEuc_TR"
#         else:
#             PTAhigh_prim = "PTA_locNomial_TL"
#             PTAlow_prim = "PTA_locEuc_TL"
#             #randidx = np.random.permutation([0,1])
#             #PTAhigh_prim = ["PTA_locNomial_TL","PTA_locNomial_TR"][randidx]
#             #PTAlow_prim = ["PTA_locEuc_TL","PTA_locEuc_TR"][randidx]

#         analyses = [
#             {"name": "train stim Compete PTAhigh Cartesian",
#             "reg_names": ["PTA_ax", f"{PTAhigh_prim}", "gtlocEuclidean"],
#             "short_names":  ["PTA_ax","PTA_locNomial","gtlocEuclidean"]
#             },
            
#             {"name": "train stim betweenxy PTA low",
#             "reg_names": [f"betweenxy_{PTAlow_prim}"],
#             "short_names": ["PTA_locEuc"]
#             },
#             {"name": "train stim betweenxy PTA high",
#             "reg_names": [f"betweenxy_{PTAhigh_prim}"],
#             "short_names": ["PTA_locNomial"]
#             }
            
#         ]
#         stim_dict = subeval_df.copy().reset_index(drop=True).to_dict('list')
#         submodelrdm = ModelRDM(
#             stimid    = stim_dict["stim_id"],
#             stimgtloc = np.vstack([stim_dict["stim_x"],stim_dict["stim_y"]]).T,
#             stimfeature = np.vstack([stim_dict["stim_color"],stim_dict["stim_shape"]]).T,
#             stimgroup = stim_dict["stim_group"],
#             sessions = stim_dict["stim_session"],
#             nan_identity = True,
#             splitgroup  =  False
#         )
#         #print(submodelrdm.models.keys())

#         for analysis_dict in analyses:
#             reg_names, short_names = analysis_dict["reg_names"], analysis_dict["short_names"]
#             reg_vals  = [submodelrdm.models[k] for k in reg_names]
#             reg_estimator  = MultipleRDMRegression(subeval_X,
#                                                 modelrdms=reg_vals,
#                                                 modelnames=short_names,
#                                                 rdm_metric="correlation")
#             reg_estimator.fit()
#             res_df = pd.DataFrame(
#                 {"modelrdm":   reg_estimator.modelnames,
#                 "coefficient": reg_estimator.result}
#             ).assign(
#             subid=subid,
#             pgroi=pgroi,
#             fitPScat = subfitPScat,
#             analysis = analysis_dict["name"],
#             adjR2 = reg_estimator.adjR2,
#             R2    = reg_estimator.R2
#             )
#             ROIRSA_PTA_cvres.append(res_df)
        
            

# dump({"RSA":ROIRSA_PTA_cvres,
#       "fit_PS":sub_cs_obs_fit,
#       "eval_PS":sub_cs_obs_eval,
#       #"fit_PSperm":sub_cs_perms_fit,
#       #"eval_PSperm":sub_cs_perms_eval,
#       "eval_dataXs":subevaldata_Xs,
#       "eval_dataDfs":subevaldata_dfs,
#       "fit_PScat":fit_subrep_category,
#       "eva_PScat":eval_subrep_category},
#       os.path.join(ROIRSAdir,"ROI_PTARSA_CV.pkl"))        
            

# savedop = load(os.path.join(ROIRSAdir,"ROI_PTARSA_cv.pkl"))
# ROIRSA_PTA_cvres = savedop["RSA"]
# ROIPS_PTA_cvres  = savedop["PS"]


"""
theoretical_tl = fitPS_df[["training_axset","training_axlocTL"]].to_numpy()
theoretical_tR = fitPS_df[["training_axset","training_axlocTR"]].to_numpy()
theoretical_ortho = fitPS_df[["stim_x","stim_y"]].to_numpy()
randomvecs = np.random.random((4,500))

PS_tl = cal_PS(theoretical_tl,fitPS_df)
PS_tr = cal_PS(theoretical_tR,fitPS_df)
PS_ortho = cal_PS(theoretical_ortho,fitPS_df)
PS_random = cal_PS(randomvecs,fitPS_df)

tl_permutations = parallel(delayed(compute_shuffle)(theoretical_tl,fitPS_df[["stim_x","stim_y"]].to_numpy(),fit_pair,p) for p in range(n_perm))
tr_permutations = parallel(delayed(compute_shuffle)(theoretical_tR,fitPS_df[["stim_x","stim_y"]].to_numpy(),fit_pair,p) for p in range(n_perm))
ortho_permutations = parallel(delayed(compute_shuffle)(theoretical_ortho,fitPS_df[["stim_x","stim_y"]].to_numpy(),fit_pair,p) for p in range(n_perm))
randomvecs_permutations = parallel(delayed(compute_shuffle)(randomvecs,fitPS_df[["stim_x","stim_y"]].to_numpy(),fit_pair,p) for p in range(n_perm))

fig,axes = plt.subplots(4,1)
sns.histplot(tl_permutations,color="grey",ax=axes[0],alpha=0.2)
sns.histplot(tr_permutations,color="grey",ax=axes[1],alpha=0.2)
sns.histplot(ortho_permutations,color="grey",ax=axes[2],alpha=0.2)
sns.histplot(randomvecs_permutations,color="grey",ax=axes[3],alpha=0.2)
axes[0].axvline(x=PS_tl,color="black")
axes[1].axvline(x=PS_tr,color="black")
axes[2].axvline(x=PS_ortho,color="black")
axes[3].axvline(x=PS_random,color="black")


      
"""