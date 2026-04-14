"""
This script generates the null distribution for the within-task RSA analysis. \n
It shuffles the stimulus locations within each participant and each task, then runs the same RSA analysis as in rsa_withintask.py.
It saves the results for each ROI, each participant, each task, each within-task RSA analysis for each shuffle iteration.
A group null can be computed by averaging across participants for each shuffle iteration.

In the current setup:
1) we used the same seed for all participants, which is more stringent than shuffling each participant independently.
2) we used the "factorised" shuffling method, which shuffles the x and y coordinates independently. \n
   for example, if the original mapping is that red-green-blue indicates -1,0,1 on x-axis and square-circle-triangle indicates -1,0,1 on y-axis, \n
   then the full factorial combinations of colours and shapes indicate the 9 locations: \n
         red-square (-1,-1), red-circle (-1,0), red-triangle (-1,1) \n
         green-square (0,-1), green-circle (0,0), green-triangle (0,1) \n
         blue-square (1,-1), blue-circle (1,0), blue-triangle (1,1) \n
    shuffling will generate new mappings like green-blue-red for x-axis and triangle-square-circle for y-axis, thus the new locations will be: \n
         red-square (0,1), red-circle (0,0), red-triangle (0,-1) \n
         green-square (1,1), green-circle (1,0), green-triangle (1,-1) \n
         blue-square (-1,1), blue-circle (-1,0), blue-triangle (-1,-1) \n
    This method preserves the factorised structure of the task and stimuli and is therefore more stringent than shuffling x-y pairs together

"""
import numpy as np
import scipy
import scipy.spatial
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import json
from copy import deepcopy
import glob
from joblib import Parallel, delayed, cpu_count, dump,load


from zpyhelper.filesys import checkdir
from zpyhelper.MVPA.rdm import upper_tri, compute_rdm, lower_tri
from zpyhelper.MVPA.preprocessors import split_data,scale_feature,concat_data,kabsch_algorithm
from zpyhelper.MVPA.estimators import MultipleRDMRegression, PatternCorrelation

from typing import Union
import sys
import os


project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
study_scripts   = os.path.join(project_path,'scripts','Exp1_fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')
sys.path.append(project_path)

from scripts.Exp1_fmri.multivariate.modelrdms import ModelRDM
from scripts.Exp1_fmri.multivariate.pirateOMutils import shuffle_locations

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Load data
study_scripts = os.path.join(project_path,'scripts','Exp1_fmri')
with open(os.path.join(study_scripts,'pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    nongeneralizers = pirate_defaults['participants']["nongeneralizerids"]
    generalizers    = pirate_defaults['participants']["generalizerids"]
print(f"N_participants = {len(subid_list)}")

roi_data = load(os.path.join(ROIRSAdir,"roi_data_4r.pkl"))
rois = list(roi_data.keys())

shuffle_method = "factorised" # the method to shuffle locations, see pirateOMutils.shuffle_locations for options
share_seeds = True # whether to use the same seed across participants for each iteration
withintaskrsadir = os.path.join(ROIRSAdir,"withintaskrsa")
shuffled_dir = os.path.join(withintaskrsadir,f"{shuffle_method}_{'sharedseeds' if share_seeds else 'diffseeds'}_null")
n_shuffles = 10000

flag_compute_meanandshuffle = False
flag_run_analysis = False
flag_plot_results = True


if flag_compute_meanandshuffle:
    # compute the mean activation pattern for each stimulus across all runs to save memory
    mean_roi_data = {"navigation":{}, "localizer":{}}
    tmpsubdf_dict = {"navigation":{}, "localizer":{}}
    for roi in rois:
        mean_roi_data["navigation"][roi] = []
        mean_roi_data["localizer"][roi] = []
        tmpsubdf_dict["navigation"][roi] = []
        tmpsubdf_dict["localizer"][roi] = []
        for subdata in roi_data[roi]:
            preprocedX = deepcopy(subdata["preprocX"])
            stimdf = subdata["stimdf"]

            navi_filter = stimdf.taskname.to_numpy() == "navigation"
            lzer_filter = stimdf.taskname.to_numpy() == "localizer"

            task_filters = {
                "navigation": navi_filter,
                "localizer": lzer_filter
            }
            
            for taskname, task_filter in task_filters.items():
                final_filter = task_filter
                #average across sessions
                curr_X = np.mean(split_data(X      = preprocedX[final_filter,:],
                                            groups = stimdf[final_filter].copy().stim_session.to_numpy()),
                                axis=0)
                #select stimdf using only one session because we did averaging
                task_df = stimdf[final_filter].copy().reset_index(drop=True)
                minsess = task_df.stim_session.min()
                curr_df = task_df[(task_df.stim_session==minsess)].copy().reset_index(drop=True)
                # reorder so that the rdm looks intepretable
                new_order = curr_df.sort_values(["training_axset","training_axlocTL","stim_x","stim_y"]).index.to_numpy()
                curr_X = curr_X[new_order,:]
                curr_df = curr_df.iloc[new_order,:].reset_index(drop=True)
                #save them
                mean_roi_data[taskname][roi].append(curr_X)
                tmpsubdf_dict[taskname][roi].append(curr_df)

    # make sure all stimdf are the same across ROIs and only keep one copy of the necessary columns for each participant and each task
    true_subdf_dict = {"navigation":[], "localizer":[]}
    for taskname in ["navigation","localizer"]:
        for roi in rois[1:]:
            for df1,df2 in zip(tmpsubdf_dict[taskname][rois[0]],tmpsubdf_dict[taskname][roi]):
                assert df1.equals(df2), "Stimulus dataframes do not match across ROIs!"

        for subid, df in zip(subid_list,tmpsubdf_dict[taskname][rois[0]]):
            stimid = df["stim_id"].to_numpy()
            stimgtloc = np.vstack([df["stim_x"],df["stim_y"]]).T
            stimfeature = np.vstack([df["stim_color"],df["stim_shape"]]).T
            stimgroup = df["stim_group"].to_numpy()
            sessions = df["stim_session"].to_numpy()
            true_subdf_dict[taskname].append(
                (subid, (
                    stimid,
                    stimgtloc,
                    stimfeature,
                    stimgroup,
                    sessions
                ))
            )

    # shuffle the locations
    shuffled_subdf_dict = [{"navigation":[], "localizer":[]} for _ in range(n_shuffles)]
    for taskname in ["navigation","localizer"]:
        for shuffle_idx in range(n_shuffles):            
            for subid, (stimid, stimgtloc, stimfeature, stimgroup, sessions) in true_subdf_dict[taskname]:
                if share_seeds:
                    # set the same seed across participants for each iteration
                    randomseed = shuffle_idx
                else:
                    # set different seed for each participant
                    randomseed = int(subid.replace("sub",""))*n_shuffles + shuffle_idx
                shuffled_locs = shuffle_locations(
                    locations=stimgtloc,
                    shuffle_function=shuffle_method,
                    randomseed=randomseed
                )
                shuffled_subdf_dict[shuffle_idx][taskname].append(
                    (subid, (
                            stimid,
                            shuffled_locs,
                            stimfeature,
                            stimgroup,
                            sessions
                        ))
                    )
    # save the results
    checkdir(shuffled_dir)
    #save all the shuffles
    dump(shuffled_subdf_dict, os.path.join(shuffled_dir,"shuffled_subdf_dict.pkl"))

else:
    shuffled_subdf_dict = load(os.path.join(shuffled_dir,"shuffled_subdf_dict.pkl"))

#run RSA analysis
def run_analysis(roidata, subdfs, returnall=False):
    res_df_list = []
    taskrdm = {"navigation": {},
            "localizer": {}}
    taskdf = {"navigation": {},
            "localizer": {}}
    rois = roidata["navigation"].keys()
    for roi in rois:
        taskrdm["navigation"][roi],taskrdm["localizer"][roi] = [], []
        taskdf["navigation"][roi], taskdf["localizer"][roi] = [], []
        for taskname in ["navigation","localizer"]:
            for subdata,(subid,stiminfo) in zip(roidata[taskname][roi],subdfs[taskname]):
                (stimid, stimgtloc, stimfeature, stimgroup, sessions) = stiminfo
                # save neural rdm
                taskrdm[taskname][roi].append(
                    compute_rdm(
                        subdata,
                        metric="correlation"
                    )
                )
                taskdf[taskname][roi].append(pd.DataFrame({
                    "subid": subid,
                    "stim_id": stimid,
                    "stim_x": stimgtloc[:,0],
                    "stim_y": stimgtloc[:,1],
                    "stim_color": stimfeature[:,0],
                    "stim_shape": stimfeature[:,1],
                    "stim_group": stimgroup
                }))
                
                # get model rdms
                submodelrdm = ModelRDM(
                    stimid    = stimid,
                    stimgtloc = stimgtloc,
                    stimfeature = stimfeature,
                    stimgroup = stimgroup,
                    sessions = sessions,
                    nan_identity = False,
                    splitgroup  = True
                )

                highDreg_name = "feature"
                lowDreg_name  = "gtlocEuclidean"
                prim_reg_names = [lowDreg_name,highDreg_name]        


                if taskname=="navigation":
                    analyses = [
                        {"name": "all Compete Feature Cartesian",
                         "reg_names": [f"{x}" for x in prim_reg_names] + ["stimuligroup"],
                         "short_names":  prim_reg_names + ["stimuligroup"]
                        },
                        {"name": "train stim withinxy - Euc",
                        "reg_names": [f"withinxy_trainstimpairs_{lowDreg_name}"],
                        "short_names": [lowDreg_name]
                        },
                        {"name": "test stim withinxy - Euc",
                        "reg_names": [f"withinxy_teststimpairs_{lowDreg_name}"],
                        "short_names": [lowDreg_name]
                        },

                        {"name": "test stim Compete high low D",
                        "reg_names": [f"teststimpairs_{x}" for x in prim_reg_names],
                        "short_names":  prim_reg_names
                        }
                        
                    ]
                else:
                    analyses = [
                        {"name": "all Compete Feature Cartesian",
                        "reg_names": [f"{x}" for x in prim_reg_names],
                        "short_names":  prim_reg_names
                        }
                    ]
                
                for analysis_dict in analyses:
                    reg_names = [k for k in analysis_dict["reg_names"] if k in submodelrdm.models.keys()]
                    short_names = [sn for sn,k in zip(analysis_dict["short_names"],analysis_dict["reg_names"]) if k in submodelrdm.models.keys()]
                    reg_vals  = [submodelrdm.models[k] for k in reg_names]
                    n_regs = len(reg_vals)
                    if n_regs == 0:
                        print(f"Skipping {roi} - {subid} - {taskname} - {analysis_dict['name']} because no regressor found")
                        continue
                    elif n_regs == 1:
                        reg_estimator = PatternCorrelation(subdata,
                                                            modelrdms=reg_vals,
                                                            modelnames=short_names,
                                                            rdm_metric="correlation",
                                                            return_cpd=False)
                    else:
                        reg_estimator  = MultipleRDMRegression(subdata,
                                                            modelrdms=reg_vals,
                                                            modelnames=short_names,
                                                            rdm_metric="correlation",
                                                            return_cpd=True)
                    reg_estimator.fit()
                    res_df = pd.DataFrame(
                        {"modelrdm":   reg_estimator.modelnames,
                         "coefficient": reg_estimator.result[:n_regs],
                         "CPD": reg_estimator.result[n_regs:] if n_regs>1 else [np.nan]*n_regs
                        }
                    ).assign(
                    roi = roi,
                    subid=subid,
                    taskname = taskname,
                    analysis = analysis_dict["name"],
                    adjR2 = reg_estimator.adjR2 if len(reg_vals)>1 else np.nan,
                    R2    = reg_estimator.R2 if len(reg_vals)>1 else np.nan
                    )
                    res_df_list.append(res_df)
    res_df = pd.concat(res_df_list,axis=0).reset_index(drop=True)    
    if returnall:
        return res_df, taskrdm, taskdf
    else:
        return res_df

if flag_run_analysis:
    # run the analysis on the true data
    res_df, taskrdm, taskdf = run_analysis(mean_roi_data, true_subdf_dict, True)
    res_df["subgroup"] = res_df.subid.apply(
        lambda x: "Generalizer" if x in generalizers else "nonGeneralizer"
    )
    # save true results
    res_df.to_csv(os.path.join(withintaskrsadir,"rsa_withintask.csv"),index=False)
    dump({"taskrdm": taskrdm,"taskdf": taskdf},     
        os.path.join(withintaskrsadir,"withintask_rdms.pkl"))

    # run the analysis on the shuffled data and save results
    null_res_dfs = Parallel(n_jobs=cpu_count())(delayed(run_analysis)(mean_roi_data, shuffled_subdfs,False) for shuffled_subdfs in shuffled_subdf_dict)
    for shuffle_idx, null_res_df in enumerate(null_res_dfs):
        null_res_df["subgroup"] = null_res_df.subid.apply(
            lambda x: "Generalizer" if x in generalizers else "nonGeneralizer"
        )
        null_res_df = null_res_df.assign(shuffle_idx=shuffle_idx)
        null_res_df.to_csv(os.path.join(shuffled_dir,f"rsa_withintask_null_shuffle{shuffle_idx+1:05d}.csv"),index=False)


if flag_plot_results:
    # plot the results comparing to null distribution and show the uncorrected pvalues
    # plot config
    participantgrouphex = {'Generalizer': '#029e73', 'nonGeneralizer': '#cc78bc'}
    roi_shortnames = dict(zip(
        rois,
        [roi.split("_")[0] for roi in rois]
    ))

    # Load Null results
    null_res_dfs = [pd.read_csv(os.path.join(shuffled_dir,f"rsa_withintask_null_shuffle{shuffle_idx+1:05d}.csv")) for shuffle_idx in range(n_shuffles)]
    null_res_df = pd.concat(null_res_dfs,axis=0).reset_index(drop=True)
    null_res_df.roi = null_res_df.roi.map(roi_shortnames)
    # compute the null for each permutation
    null_summary = null_res_df.groupby(
                        ["shuffle_idx","roi","taskname","analysis","modelrdm","subgroup"]
                    )[["coefficient","CPD"]].mean().reset_index()

    # load the true results
    res_df = pd.read_csv(os.path.join(withintaskrsadir,"rsa_withintask.csv"))
    res_df = res_df[res_df.analysis.isin(null_summary.analysis.unique())].copy()
    res_df.roi = res_df.roi.map(roi_shortnames)

    for tname, taskdf in res_df.groupby("taskname"):
        for aname, analysisdf in taskdf.groupby("analysis"):
            yvar = "coefficient"
            xvar = "roi"
            curr_pdf = analysisdf.copy().reset_index(drop=True)            
            curr_pdf.roi = pd.Categorical(curr_pdf.roi, categories=["V1","PPC","HPC","vmPFC"],ordered=True)
            curr_pdf[xvar] = pd.Categorical(curr_pdf[xvar],ordered=True)
            assert curr_pdf.shape[0]>0, f"No data"

            curr_null_df = null_summary[
                    (null_summary.analysis==aname) &
                    (null_summary.taskname==tname)
                ].copy()
            curr_null_df.roi = pd.Categorical(curr_null_df.roi, categories=["V1","PPC","HPC","vmPFC"],ordered=True)
            curr_null_df[xvar] = pd.Categorical(curr_null_df[xvar],ordered=True)
            assert curr_null_df.shape[0]>0, "No null data found!"
                
            unique_modelrdms = curr_pdf.modelrdm.unique()
            if len(unique_modelrdms)>1:
            
                atitle = f"{tname} {aname}"
            
                gs_groupnull = sns.catplot(data=curr_null_df,
                            x=xvar,y=yvar,
                            #hue="subgroup",palette=participantgrouphex,
                            color="grey",
                            row="subgroup",
                            col="modelrdm",
                            kind="violin",
                            alpha=0.5,
                            split=True,
                            inner =None,
                            height=3.7,aspect=.85
                            ).refline(y=0).set_titles("{row_name}\n{col_name}")
                gs_groupnull.figure.suptitle(atitle,y=1.03)
                gs_groupnull._legend.remove()
                gs_groupnull.tight_layout(w_pad=3)
    
                for ax in gs_groupnull.axes.flatten():
                    curr_modelrdm = ax.get_title().split("\n")[-1]
                    print(curr_modelrdm)
                    curr_subgroup = ax.get_title().split("\n")[0]
                    sns.pointplot(
                            curr_pdf[
                            (curr_pdf.analysis==aname) &
                            (curr_pdf.taskname==tname) &
                            (curr_pdf.modelrdm==curr_modelrdm) &
                            (curr_pdf.subgroup.str.lower()==curr_subgroup.lower())
                        ].copy(),
                            x = "roi",
                            y= yvar,
                            color="black",#participantgrouphex[curr_subgroup],
                            join=False,
                            scale=0.8,
                            markers="o",
                            errorbar=None,
                            ax=ax
                        )
                    for roi in curr_pdf.roi.unique():
                        bar_null_df = curr_null_df[
                            (curr_null_df.modelrdm==ax.get_title().split("\n")[-1]) &
                            (curr_null_df.subgroup.str.lower()==ax.get_title().split("\n")[0].lower()) &
                            (curr_null_df.roi==roi)
                        ].copy()
                        assert bar_null_df.shape[0]==n_shuffles, f"expected {n_shuffles} null samples! got {bar_null_df.shape[0]}"

                        bar_true_df = curr_pdf[
                            (curr_pdf.analysis==aname) &
                            (curr_pdf.taskname==tname) &
                            (curr_pdf.modelrdm==curr_modelrdm) &
                            (curr_pdf.subgroup.str.lower()==curr_subgroup.lower()) &
                            (curr_pdf.roi==roi)
                        ].copy()
                        assert bar_true_df.shape[0]>0, "No null data found!"
                        mean_vals = bar_true_df[yvar].to_numpy().mean()
                        sem_vals   = scipy.stats.sem(bar_true_df[yvar].to_numpy())
                        pval = 1-np.mean(mean_vals>bar_null_df[yvar].to_numpy())
                        pval = pval*2 # two-tailed
                        print(f"{roi} - {curr_modelrdm} - {curr_subgroup} - mean: {mean_vals:.4f}, sem: {sem_vals:.4f}, pval: {pval:.4f}")
                        # plot error bar and significance
                        if pval<0.001:
                            sig_symbol = "***"
                        elif pval<0.01:
                            sig_symbol = "**"
                        elif pval<0.05:
                            sig_symbol = "*"
                        else:
                            sig_symbol = ""

                        
                        # ax.errorbar(
                        #     x = list(curr_pdf.roi.cat.categories).index(roi),
                        #     y=mean_vals,
                        #     yerr=sem_vals,
                        #     fmt='o',
                        #     color="black",#participantgrouphex[curr_subgroup],
                        #     capsize=5
                        # )
                        ax.text(
                            x = list(curr_pdf.roi.cat.categories).index(roi),
                            y=mean_vals + sem_vals + 0.01,
                            s=sig_symbol,
                            color="black",#participantgrouphex[curr_subgroup],
                            ha='center',
                            va='bottom',
                            fontsize=12
                        )
            else:
                pass
