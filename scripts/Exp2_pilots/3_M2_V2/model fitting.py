import numpy as np
import scipy.stats
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import scipy
import itertools
project_dir = "E:\pirate_fmri\Analysis"
import sys
sys.path.append(project_dir)
from src.utils.composition_modelfit import *
from scipy.optimize import minimize
from joblib import Parallel,delayed,dump,load

study_dir = r"E:\pirate_fmri\Analysis\data\Exp2_pilots\3_M2_V2"
data_dir = os.path.join(study_dir,"data","json")
param_dir = os.path.join(study_dir,"data","param")

org_data = pd.read_csv(os.path.join(study_dir,"task_data.csv"))
valid_exptid = np.unique(org_data.subid)

# rescale stim_xy and respone
range_stimxy = [np.max(org_data.stim_x) - np.min(org_data.stim_x),
                np.max(org_data.stim_y) - np.min(org_data.stim_y)]
org_data["stim_xO"] = org_data.stim_x
org_data["stim_yO"] = org_data.stim_y
org_data["resp_xO"] = org_data.resp_x
org_data["resp_yO"] = org_data.resp_y
org_data["stim_x"]  = [np.round(2*a/range_stimxy[0],decimals = 3) for a in org_data.stim_xO.to_numpy()]
org_data["stim_y"]  = [np.round(2*a/range_stimxy[1],decimals = 3) for a in org_data.stim_yO.to_numpy()]
org_data["resp_x"]  = [np.round(2*a/range_stimxy[0],decimals = 3) for a in org_data.resp_xO.to_numpy()]
org_data["resp_y"]  = [np.round(2*a/range_stimxy[0],decimals = 3) for a in org_data.resp_yO.to_numpy()]


putativerespdata_test = []
for id in valid_exptid:
    subdf = org_data[(org_data.subid==id)&(org_data.ctrl_ept==1)].copy().reset_index(drop=True)
    
    # find the final block of each map in test
    n_pretrain_permap = 8
    n_map = 2
    n_refresh_permap = 4
    test_block_idx = n_pretrain_permap*n_map+n_refresh_permap
    sub_testblock_df = subdf[(subdf.expt_session==1)&(subdf.expt_block==test_block_idx)].copy().reset_index(drop=True)
    #sub_testblock_df.groupby(["subid","expt_map","x_axisset","y_axisset","stim_group","istraining","stim_id","stim_x","stim_y"])[["resp_x","resp_y"]].mean().reset_index()
    # the data frame used to derive putative response
    sub_trainingresp=subdf[(subdf.expt_session==1)&(subdf.expt_block==test_block_idx-1)].copy().reset_index(drop=True)
    sub_resp=sub_testblock_df.copy()
    
    putative_xs = compose_pattern_from_reference(sub_resp[["stim_x"]].to_numpy(),
                                sub_trainingresp[["resp_x"]].to_numpy(),
                                sub_trainingresp[["stim_x"]].to_numpy(),
                                source_controlfeatures=sub_resp[["x_axisset"]].to_numpy(),
                                reference_controlfeatures=sub_trainingresp[["x_axisset"]].to_numpy()).flatten()
    putative_ys = compose_pattern_from_reference(sub_resp[["stim_y"]].to_numpy(),
                                sub_trainingresp[["resp_y"]].to_numpy(),
                                sub_trainingresp[["stim_y"]].to_numpy(),
                                source_controlfeatures=sub_resp[["y_axisset"]].to_numpy(),
                                reference_controlfeatures=sub_trainingresp[["y_axisset"]].to_numpy()).flatten()
    sub_resp["putative_x"] = [truex if istraining else putativex for putativex,truex,istraining in zip(putative_xs,sub_resp.stim_x.to_numpy(),sub_resp.istraining.to_numpy())]
    sub_resp["putative_y"] = [truey if istraining else putativey for putativey,truey,istraining in zip(putative_ys,sub_resp.stim_y.to_numpy(),sub_resp.istraining.to_numpy())]
    sub_resp["map_axes"] = [f"{x}_{y}" for x,y in sub_resp[["x_axisset","y_axisset"]].to_numpy()]
    putativerespdata_test.append(sub_resp)

putativerespdata_test_df = pd.concat(putativerespdata_test).reset_index(drop=True)
putativerespdata_test_df["iscenter"] = pd.Categorical([x*y==0 for x,y in putativerespdata_test_df[["stim_x","stim_y"]].to_numpy()]).rename_categories(
    {True:"center",False:"noncenter"}
)

nrow=8
ncol=5
fig,axes = plt.subplots(nrow,ncol,figsize=(5*4,8*4))
for j,id in enumerate(valid_exptid):
    subdf = putativerespdata_test_df[(putativerespdata_test_df.subid == id)&(putativerespdata_test_df.iscenter == "noncenter")].copy().reset_index(drop=True)
    sns.scatterplot(subdf,x="putative_x",y="putative_y",c="blue",ax=axes.flatten()[j],label="putative2D")
    sns.scatterplot(subdf,x="resp_x",y="resp_y",c="orange",ax=axes.flatten()[j],label="actual response")
    if j == len(valid_exptid)-1:
        figlgdhandles, figlgdlabels = axes.flatten()[j].get_legend_handles_labels()
    axes.flatten()[j].get_legend().remove()
    axes.flatten()[j].set_title(id)
    axes.flatten()[j].set_xlabel("x")
    axes.flatten()[j].set_xlabel("y")
if j<axes.size-1:
    [axes.flatten()[j].remove() for j in np.arange(axes.size-1-j)+len(valid_exptid)]
fig.legend(figlgdhandles,figlgdlabels,loc="upper center", ncol=2, bbox_to_anchor=(1,1.04))
fig.tight_layout()
fig.savefig(os.path.join(study_dir,"putative and actual resp.png"))

################# RUN MODEL FITTING ##############################
run_fit = False
# arena parameters
arena_r = (5 + (1.42*60/53*1.1/2))*(2/range_stimxy[0])
err_tol = 1.05*60/53 *(2/range_stimxy[0]) # take into account the pirate size

#double_check_subs = ['H4wAVOPtFkAT','YHFTln2WomAo']  #'G8yNoK8WMiwt','l5kKVclV7Cy9','UQoUjJ6BxXdi',
#valid_exptid = double_check_subs
outout_fn = "fitted_bhavmodel_parameters_tvsplit_centerality"
if run_fit:
    # models to fit
    model_xystrategies = { # in the form of xstrategy-ystrategy
            "compositional": ["putative-putative"],
            "1Dx-center":["putative-bias"],
            "1Dy-center":["bias-putative"],
            "1D-center":["putative-bias","bias-putative"],
            "1D-random":["putative-random","random-putative"]
        }
    model_paramnames = {
        "noncompo-random":[],
        "compositional": ["sigma_x", "sigma_y", "lapse_rate"], # "betax", "betay", 
        "1Dx-center":    ["sigma_x", "sigma_biasy", "lapse_rate"],
        "1Dy-center":    ["sigma_y", "sigma_biasx", "lapse_rate"],
        "1D-center":     ["sigma_x", "sigma_y", "sigma_biasx", "sigma_biasy", "lapse_rate"],
        "1D-random":     ["sigma_x", "sigma_y", "lapse_rate"]
    }
    #beta_bounds = (0.1,arena_r)
    #mu_bounds = (-arena_r,arena_r)
    lapse_bounds = (0,0.1)

    sigma_bounds = (1e-20,err_tol) 
    sigmabias_bounds = (1e-20,err_tol)     
    model_bounds = {
        "compositional": [sigma_bounds,sigma_bounds,lapse_bounds,lapse_bounds],
        "1Dx-center":    [sigma_bounds,sigmabias_bounds,lapse_bounds,lapse_bounds],
        "1Dy-center":    [sigma_bounds,sigmabias_bounds,lapse_bounds,lapse_bounds],
        "1D-center":[sigma_bounds,sigma_bounds,sigmabias_bounds,sigmabias_bounds,lapse_bounds,lapse_bounds],
        "1D-random":[sigma_bounds,sigma_bounds,lapse_bounds,lapse_bounds]
    }
    unique_param_names = np.unique(sum(list(model_paramnames.values()),[]))

    fixedparams = {     
                    "noncompo-random":{},
                    "compositional":  {"betax":1,"betay":1},
                    "1Dx-center":     {"betax":1,"betay":np.nan,"biasy":0},
                    "1Dy-center":     {"betax":np.nan,"betay":1,"biasx":0},
                    "1D-center":      {"betax":1,"betay":1,"biasx":0, "biasy":0},
                    "1D-random":      {"betax":1,"betay":1,"biasx":0, "biasy":0} # 
                }
    all_param_names = np.array(list(unique_param_names) + ["betax", "betay","biasx","biasy","lapse_rate_2D"])


    fit_param = []
    for j,id in enumerate(valid_exptid):
        subdf = putativerespdata_test_df[(putativerespdata_test_df.subid == id)&(putativerespdata_test_df.stim_group=="validation")&(putativerespdata_test_df.expt_map.isin([0,1]))].copy().reset_index(drop=True)
        #trainstim_df = putativerespdata_test_df[(putativerespdata_test_df.subid == id)&(putativerespdata_test_df.stim_group=="training")&(putativerespdata_test_df.expt_map.isin([0,1]))].copy().reset_index(drop=True)
        #mu_respx = np.mean(trainstim_df[trainstim_df.stim_x==0].resp_x)
        #mu_respy = np.mean(trainstim_df[trainstim_df.stim_y==0].resp_y)
        #fixedparams["1D-center"].update({"biasx":mu_respx, "biasy":mu_respy})
        
        fit_data = subdf[["resp_x","resp_y","putative_x","putative_y"]].to_numpy()
        
        #estimate starting value for optimisation
        bx, ix, _, _, _  = scipy.stats.linregress(fit_data[:,0], fit_data[:,2])
        by, iy, _, _, _  = scipy.stats.linregress(fit_data[:,1], fit_data[:,3])  
        bx,by = 1,1
        sig_x = np.power(fit_data[:,0] - bx*fit_data[:,2],2).mean()
        sig_y = np.power(fit_data[:,1] - by*fit_data[:,3],2).mean()

        sigbias_x = np.power(fit_data[:,0] - 0,2).mean()
        sigbias_y = np.power(fit_data[:,1] - 0,2).mean()

        model_x0 = {
            "compositional": [sig_x,sig_y,0.05,0.05],
            "1Dx-center":    [sig_x,sigbias_y,0.05,0.0],
            "1Dy-center":    [sig_y,sigbias_x,0.05,0.05],
            "1D-center":     [sig_x,sig_y,sigbias_x,sigbias_y,0.05,0.05],
            "1D-random":     [sig_x,sig_y,0.05,0.05]
        }

        all_model_fit = []
        for mname, xystrategies in model_xystrategies.items():
            print(f"\nsub{j+1}/{np.size(valid_exptid)}: {id} {mname}")

            #fit_output = multi_start_optimisation(compute_llr_across_trials,
            #                         bounds=model_bounds[mname],
            #                         args=(model_paramnames[mname],xystrategies,fit_data,fixedparams[mname]),
            #                         Ns=20,n_jobs=10)
            fit_output = wise_start_optimisation(compute_llr_across_trials,
                                    bounds=model_bounds[mname],
                                    args=(model_paramnames[mname],xystrategies,fit_data,arena_r,fixedparams[mname]),
                                    x0=model_x0[mname],
                                    minimize_args={})
            print(f"  fitted {model_paramnames[mname]} = {fit_output[0]}, nll = {fit_output[1]}")
            all_model_fit.append(fit_output)
        fit_param.append(all_model_fit)
    dump({"subids":valid_exptid,"fit_param":fit_param,
          "fixedparams":fixedparams,"model_paramnames":model_paramnames,
          "model_bounds":model_bounds,"model_xystrategies":model_xystrategies},os.path.join(study_dir,f"{outout_fn}.pkl"))


################# CHECK MODEL FIT ##############################
check_fit = True
if check_fit:
    fitted_data  = load(os.path.join(study_dir,f"{outout_fn}.pkl"))
    fixedparams  = fitted_data["fixedparams"]
    valid_exptid = fitted_data["subids"]
    fit_param    = fitted_data["fit_param"]
    fitted_models = list(model_xystrategies.keys())
    model_selection_dfs = []
    model_param_dfs = []
    keep_models = fitted_models+["noncompo-random"]
    for jid,(id,fit_output) in enumerate(zip(valid_exptid,fit_param)):
        print(f"sub{jid+1}/{len(valid_exptid)}: {id}")
        fitted_param = dict(zip(
            fitted_models+["noncompo-random"],
            [o[0] for o in fit_output]+[[]]
        ))

        fitted_nll = dict(zip(
            fitted_models+["noncompo-random"],
            [o[1] for o in fit_output]+[[]]
        ))

        model_xystrategies["noncompo-random"] = ["random-random"]
        # model_xystrategies = { # in the form of xstrategy-ystrategy
        #     "noncompo-random":["random-random"],
        #     "compositional": ["putative-putative"],
        #     #"1Dx-center":["putative-bias"],
        #     #"1Dy-center":["bias-putative"],
        #     "1D-center":["putative-bias","bias-putative"],
        #     "1D-random":["putative-random","random-putative"]
        # }

        sub_df = putativerespdata_test_df[putativerespdata_test_df.subid == id].copy().reset_index(drop=True)
        
        df_split_criteria = { 
            "trainingmaps validation":          (sub_df.expt_map.isin([0,1]))&(sub_df.stim_group == "validation"), # dataset that is used to fit the parameter
            "trainingmaps training-and-center": (sub_df.expt_map.isin([0,1]))&(sub_df.iscenter == "center"),       # for sanity check
            "trainingmaps noncenter-test":      (sub_df.expt_map.isin([0,1]))&(sub_df.iscenter == "noncenter")&(sub_df.stim_group == "test"),    #
            "crossmaps center-test":            (sub_df.expt_map.isin([2,3]))&(sub_df.iscenter == "center"),
            "crossmaps noncenter-test":         (sub_df.expt_map.isin([2,3]))&(sub_df.iscenter == "noncenter"),
        }
        df_splits = {}
        data_splits = {}
        for k,v in df_split_criteria.items():
            df_splits[k] = sub_df[v].copy().reset_index(drop=True)
            data_splits[k] = df_splits[k][["resp_x","resp_y","putative_x","putative_y"]].to_numpy()
        
        data_nll = {}
        data_BIC = {}
        data_R2  = {}
        data_slope = {}
        n_random_sample = 500
        for sgname, ds in data_splits.items():
            data_nll[sgname] = {}
            data_BIC[sgname] = {}
            data_R2[sgname] = {}
            data_slope[sgname] = {}
            for mname in keep_models:
                mparam = fitted_param[mname]
                data_nll[sgname][mname] = compute_llr_across_trials(mparam,model_paramnames[mname],model_xystrategies[mname],ds,arena_r,fixedparams[mname])
                if np.logical_and(mname != "noncompo-random", sgname == "trainingmaps validation"):
                    returned_nll = fitted_nll[mname]
                    assert np.round(data_nll[sgname][mname],4) == np.round(returned_nll,4)
                n_trials = df_splits[sgname].shape[0]
                data_BIC[sgname][mname] = BIC(data_nll[sgname][mname],n_trials,len(mparam))

                param_dict = dict(zip(model_paramnames[mname],fitted_param[mname]))
                param_dict.update(fixedparams[mname])
                null_param_dict = dict(zip(all_param_names[[x not in param_dict.keys() for x in all_param_names]],
                                        [np.nan]*sum([x not in param_dict.keys() for x in all_param_names])))
                param_dict.update(null_param_dict)
                # predicted_resp is of shape (n_random_sample,2,ds.shape[0])
                predicted_resp = sample_response(param_dict, model_xystrategies[mname], ds, arena_r, sample_size=n_random_sample,seed=None)
                true_resp = ds[:,[0,1]]
                slp = [calculate_regression(predicted_resp[ksample,:,:].T, true_resp) for ksample in range(n_random_sample)]
                r2 = [calculate_r_squared_bivariate(predicted_resp[ksample,:,:].T, true_resp) for ksample in range(n_random_sample)]
                data_R2[sgname][mname] = np.mean(r2)
                data_slope[sgname][mname] = np.mean(slp)

        modelnames = keep_models
        model_param_df_dict = {"modelnames":modelnames}
        for pname in unique_param_names:
            model_param_df_dict[pname] = [fitted_param[mname][model_paramnames[mname].index(pname)] if pname in model_paramnames[mname] else np.nan for mname in modelnames]
        
        for sgname in data_nll.keys():
            n_trials = df_splits[sgname].shape[0]            
            model_param_df_dict[f"{sgname} BIC"]          = [data_BIC[sgname][mname] for mname in modelnames]
            model_param_df_dict[f"{sgname} R2"]           = [data_R2[sgname][mname] for mname in modelnames]
            model_param_df_dict[f"{sgname} slope(true~pred)"] = [data_slope[sgname][mname] for mname in modelnames]
            model_param_df_dict[f"{sgname} meantrialNLL"] = [data_nll[sgname][mname]/n_trials for mname in modelnames]
            bmodel = list(data_BIC[sgname].keys())[np.argmin(list(data_BIC[sgname].values()))]
            model_param_df_dict[f"{sgname} bestmodel"] = [bmodel] * len(modelnames)

        model_param_dfs.append(
            pd.DataFrame(model_param_df_dict).assign(subid=id)
        )

    model_param_df = pd.concat(model_param_dfs,axis=0).reset_index(drop=True)

    model_param_df["iscompositional"] = pd.Categorical(
        model_param_df["trainingmaps validation bestmodel"] == "compositional"
    ).rename_categories({True: 'Compositional', False: 'nonCompositional'})
    model_param_df["bestmodel_consistency"] = model_param_df["trainingmaps validation bestmodel"] == model_param_df["crossmaps noncenter-test bestmodel"]
    
    model_param_df.to_csv(os.path.join(study_dir,"model_fit_result.csv"))
    compositional_ids = np.unique(model_param_df[model_param_df.iscompositional=="Compositional"].subid)
    remove_sub = ["RcwaTugr8Fiz","FZy8PSATlIm3","SyZeCxfhSm2n","bHAKEOQOX39l"]
    keep_sub = [id for id in valid_exptid if id not in remove_sub]

    compositional_ids = [id for id in compositional_ids if id in keep_sub]
    noncompositional_ids = [id for id in keep_sub if id not in compositional_ids]
    sorted(compositional_ids,key=str.casefold)

    best_fit_df = model_param_df[(model_param_df.modelnames==model_param_df["trainingmaps validation bestmodel"])&(~model_param_df.subid.isin(remove_sub))].copy().reset_index()

    x_dsname = 'trainingmaps validation' 
    compare_dsnames = ['trainingmaps training-and-center',
                        'trainingmaps noncenter-test',
                        'crossmaps center-test',
                        'crossmaps noncenter-test']
    compare_metrics = ["meantrialNLL","R2"]

    fig_modelfit, axes_modelfit = plt.subplots(len(compare_metrics),len(compare_dsnames),figsize=(12,6))
    for j, mtc in enumerate(compare_metrics):
        for k, y_dsname in enumerate(compare_dsnames):
            sns.scatterplot(best_fit_df,
                            x = f"{x_dsname} {mtc}",
                            y = f"{y_dsname} {mtc}",
                            hue="iscompositional",
                            ax = axes_modelfit[j,k])
            axes_modelfit[j,k].set_xlabel(f"fitting dataset:\n{x_dsname} stimuli")
            axes_modelfit[j,k].set_ylabel(f"CV dataset:\n{y_dsname} stimuli")
            axes_modelfit[j,k].set_title(mtc)
            metric_col = [col for col in best_fit_df if col.endswith(mtc)]
            ax_lims = [np.min(best_fit_df[metric_col])*1.1,np.max(best_fit_df[metric_col])*1.1]
            if mtc == "R2":
                ax_lims[0] = min([0.5,ax_lims[0]])
    
            axes_modelfit[j,k].axline((ax_lims[0], ax_lims[0]), (ax_lims[1], ax_lims[1]), linewidth=2, color='r')
            if np.logical_and(j == len(compare_metrics)-1,k == len(compare_dsnames)-1):
                figlgdhandles, figlgdlabels = axes_modelfit[j,k].get_legend_handles_labels()
            
            axes_modelfit[j,k].get_legend().remove()
            axes_modelfit[j,k].set_aspect(1)
            axes_modelfit[j,k].set_xlim(ax_lims)
            axes_modelfit[j,k].set_ylim(ax_lims)
    
    fig_modelfit.legend(figlgdhandles, figlgdlabels, loc='upper center',bbox_to_anchor=(0.5,1.1),
                        ncol=2)
    fig_modelfit.tight_layout()

    pparams = ['sigma_x', 'sigma_y', 'lapse_rate']
    fig_typedist, axes_typedist = plt.subplots(1,1+len(pparams),figsize=((1+len(pparams))*4,4))
    sns.histplot(best_fit_df,
                x="trainingmaps validation bestmodel",
                hue="modelnames",
                ax=axes_typedist[0])
    for k,pp in enumerate(pparams):
        sns.histplot(best_fit_df,
                x=pp,
                hue="iscompositional",
                ax=axes_typedist[k+1])
        axes_typedist[k+1].set_title(pp)
    fig_typedist.tight_layout()

    best_fit_df["sigma"] = [sx if np.isnan(sy) else sy if np.isnan(sx) else (sx+sy)/2 for sx,sy in best_fit_df[["sigma_x","sigma_y"]].to_numpy()]
    gs_sigma = sns.relplot(
        best_fit_df,
        x="sigma",
        y="trainingmaps noncenter-test meantrialNLL",
        hue="iscompositional"
        )
    for ax in gs_sigma.axes.flatten():
        ax.set_xlim([0,1.1*err_tol])
        ax.set_ylim([0,1.1*np.max(best_fit_df["trainingmaps noncenter-test meantrialNLL"])])
        #ax.axline(xy1=[0,0],xy2=[13,13],c="k")

    fig_consistency,ax_consistency = plt.subplots(1,1)
    sns.swarmplot(
        best_fit_df,
        x="iscompositional",
        y="bestmodel_consistency",
        hue="iscompositional",
        ax=ax_consistency)
    sns.move_legend(ax_consistency,
                    loc="upper center",
                    bbox_to_anchor=(0.5,1.01))
