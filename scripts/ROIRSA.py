"""
This script runs RSA analysis in anatomical ROIs

"""

import itertools
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import ModelRDM, compute_rdm, checkdir, lower_tri,upper_tri,scale_feature
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
import pandas as pd
import glob


project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'
ROIRSA_output_path = os.path.join(fmri_output_path,'ROIRSA',glm_name)
checkdir(ROIRSA_output_path)

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']

preprocess = ["smoothed5mmLSA","unsmoothedLSA"]

analysis = {
        # image-based rdm
        "sc_betweens_stimuli":   ['between_stimuli'],
        "sc_alls_stimuli":       ['stimuli'],
        # feature-based: color(x) or shape(y)
        "sc_withins_feature1d":  ['within_feature1dx','within_feature1dy'],
        "sc_betweens_feature1d": ['between_feature1dx','between_feature1dy'],
        "sc_alls_feature1d":     ['feature1dx','feature1dy'],
        # train-test
        "tt_withins_stimuligroup":  ['within_stimuligroup'],
        "tt_betweens_stimuligroup": ['between_stimuligroup'],
        "tt_alls_stimuligroup":     ['stimuligroup'],
        # map-based
        "betweens_loc2d":  ["between_loc2d"],
        "withins_loc2d":   ["within_loc2d"],
        "alls_loc2d":      ["loc2d"],
        "betweens_loc1d":  ["between_loc1dx","between_loc1dy"],
        "withins_loc1d":   ["within_loc1dx","within_loc1dy"],
        "alls_loc1d":      ["loc1dx","loc1dy"],
        # map-based plus control
        "betweens_loc2d_c":  ["between_stimuligroup",'between_feature1dx','between_feature1dy',"between_loc2d"],
        "withins_loc2d_c":   ["within_stimuligroup",'within_feature1dx','within_feature1dy',"within_loc2d"],
        "alls_loc2d_c":      ["stimuligroup",'feature1dx','feature1dy',"loc2d"],
        # random
        "betweens_loc2d":  ["between_random"],
        "withins_loc2d":   ["within_random"],
        "alls_loc2d":      ["random"],
        }
corr_rdm_names = np.unique(sum(list(analysis.values()),[]))

anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'
with open(os.path.join(project_path,'scripts','anatomical_masks.json')) as f:    
    anat_roi = list(json.load(f).keys())
laterality = ["left","right","bilateral"]

def run_ROIRSA(beta_img, mask_imgs,
               stimid,group,session,
               analyis_reg_vals, analyis_reg_names,
               correlation_rdm_vals,correlation_rdm_names,
               centering:bool=False,group_centering:bool=False):
    APD = ActivityPatternDataLoader(beta_img,mask_imgs)
    activitypattern = APD.X
    if centering:
        activitypattern = scale_feature(activitypattern,1,False)
    if group_centering:
        for k in group:
            activitypattern[np.where(group == k),:] = scale_feature(activitypattern[np.where(group == k),:],1,False)

    ## compute neural rdm
    neural_rdm = compute_rdm(activitypattern,'correlation')
    lt,idx = lower_tri(neural_rdm)    #lower triangle part excluding the diagonal
    #repeat range and filter by indices    
    c = np.repeat(np.arange(0, neural_rdm.shape[0]), neural_rdm.shape[0]).reshape(neural_rdm.shape)[idx] 
    i = np.tile(np.arange(0, neural_rdm.shape[0]), neural_rdm.shape[0]).reshape(neural_rdm.shape)[idx]
    #create DataFrame
    neuralrdmdf = pd.DataFrame({'stimidA':stimid[c], 'stimidB': stimid[i], 
                                'groupA': group[c],  'groupB': group[i], 
                                'runA': session[c],    'runB': session[i], 
                                'dissimilarity': lt}).set_index(['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB'])

    result = dict(zip(analyis_reg_vals.keys(),[[]] * len(analyis_reg_vals.keys())))
    for k,v in analyis_reg_vals.items():
        MR = MultipleRDMRegression(neural_rdm,v)
        MR.fit()
        result[k] = MR.result
    reg_df = []
    for k, v in analyis_reg_names.items():
        reg_name_dict = dict(zip(range(len(v)),v))
        res_df = pd.DataFrame(result[k]).T.rename(columns = reg_name_dict).assign(analysis=k)
        reg_df.append(res_df)

    PC_s =  PatternCorrelation(neuralrdm = neural_rdm,
                             modelrdm = correlation_rdm_vals,
                             type="spearman")
    PC_s.fit()
    PC_k =  PatternCorrelation(neuralrdm = neural_rdm,
                            modelrdm = correlation_rdm_vals,
                            type="kendall")
    PC_k.fit()
    corr_name_dict = dict(zip(range(len(correlation_rdm_names)),correlation_rdm_names))
    corr_df = pd.concat(
        [pd.DataFrame(PC_s.result).T.rename(columns = corr_name_dict).assign(analysis="spearman"),
         pd.DataFrame(PC_k.result).T.rename(columns = corr_name_dict).assign(analysis="kendall")],
         axis=0)
    
    
    return pd.concat(reg_df,axis=0),corr_df,neuralrdmdf


def modelrdm_to_df(rdm,stimid,group,session,modelname):
    lt,idx = lower_tri(rdm)
    c = np.repeat(np.arange(0, rdm.shape[0]), rdm.shape[0]).reshape(rdm.shape)[idx]
    i = np.tile(np.arange(0, rdm.shape[0]), rdm.shape[0]).reshape(rdm.shape)[idx]
    df = pd.DataFrame({'stimidA':stimid[c], 'stimidB': stimid[i], 
                       'groupA': group[c],  'groupB': group[i], 
                       'runA': session[c],    'runB': session[i], 
                        modelname: lt}).set_index(['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB'])
    return df

# preprocess - dataset - voxel selection - roi -laterality
reg_res_df = []
corr_res_df = []
for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)            
    ########################## Get Stimuli Data ##############################################
    beta_flist4r = []
    beta_flistoe = []
    fmask_flist = []
    rmask_flist = []
    pmask_flist = []
    run_stim_labels = []
    y_dict = {"id":[],
            "image":[],
            "locx":[],
            "locy":[],
            "color":[],
            "shape":[],
            "training":[]} 

    for subid in subid_list:
        print(f"retrieving data from {subid}")

        # load stimuli list
        stim_list_fn = glob.glob(os.path.join(fmri_output_path,'beh',subid,'sub*_stimlist.txt'))[0]
        stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False)
        # get stimuli id
        stim_id = np.array(stim_list['stim_id'])
        # get stimuli image
        stim_image = np.array([x.replace('.png','') for x in stim_list["stim_img"]])
        # get 2d location
        stim_locx = np.array(stim_list['stim_x'])
        stim_locy = np.array(stim_list['stim_y'])
        # get visual features
        stim_color = np.array([x.replace('.png','').split('_')[0] for x in stim_list["stim_img"]])
        stim_shape = np.array([x.replace('.png','').split('_')[1] for x in stim_list["stim_img"]])
        # get training/test stimuli classification
        stim_train = np.array(stim_list['training'])

        # build list of beta maps
        firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)

        y_dict["id"].append(stim_id)
        y_dict["image"].append(stim_image)
        y_dict["locx"].append(stim_locx)
        y_dict["locy"].append(stim_locy)
        y_dict["color"].append(stim_color)
        y_dict["shape"].append(stim_shape)
        y_dict["training"].append(stim_train)

        beta_flist4r.append(os.path.join(firstlvl_dir,'stimuli_4r.nii'))    
        beta_flistoe.append(os.path.join(firstlvl_dir,'stimuli_oe.nii'))    
        fmask_flist.append(os.path.join(firstlvl_dir,'mask.nii'))
        rmask_flist.append(os.path.join(firstlvl_dir,'reliability_mask.nii'))
        pmask_flist.append(os.path.join(firstlvl_dir,'permuted_reliability_mask.nii'))
    
    ########################## Run RSA ##############################################
    n_sess = {"fourruns":4,
              "oddeven":2}
    beta_flist = {"fourruns":beta_flist4r,
                "oddeven":beta_flistoe}
    mask_flist = {"noselection":fmask_flist,
                  "reliability_ths0":rmask_flist,
                  "permuted_rmask":pmask_flist}

    for ds_name, ds in beta_flist.items():
        for vselect in mask_flist:                     
            all_roirdm_df = []
            for j,subid in enumerate(subid_list):
                model_rdm = ModelRDM(stimid = y_dict["image"][j],
                                     stimloc = np.vstack([y_dict["locx"][j],y_dict["locy"][j]]).T,
                                     stimfeature = np.vstack([y_dict["color"][j],y_dict["shape"][j]]).T,
                                     stimgroup = y_dict["training"][j],
                                     n_session=n_sess[ds_name])
                ## model rdm to data frame
                df_inc_models =['loc2d', 'loc1dx', 'loc1dy', 'feature2d', 'feature1dx', 'feature1dy', 'stimuli', 'stimuligroup','random']
                modeldf = [modelrdm_to_df(rdm       = model_rdm.models[m],
                                          stimid    = np.tile(y_dict["id"][j],(n_sess[ds_name],)),
                                          group     = np.tile(y_dict["training"][j],(n_sess[ds_name],)),
                                          session   = np.concatenate([np.repeat(j,len(y_dict["id"][j])) for j in range(n_sess[ds_name])]),
                                          modelname = m) for m in df_inc_models]
                modeldf = modeldf[0].join(modeldf[1:])

                reg_models = {k: [model_rdm.models[m] for m in v] for k, v in analysis.items()}
                corr_rdm_vals = [model_rdm.models[m] for m in corr_rdm_names]
                for roi, lat in itertools.product(anat_roi, laterality):
                    print(f"{p} - {ds_name} - {vselect} - {subid} - {roi} = {lat}")
                    anat_mask = os.path.join(anatmaskdir,f'{roi}_{lat}.nii')
                    subroireglat_df,subroicorlat_df,subroirdmlat_df = run_ROIRSA(
                                                                beta_img = ds[j],
                                                                mask_imgs = [mask_flist[vselect][j],anat_mask], 
                                                                stimid = np.tile(y_dict["id"][j],(n_sess[ds_name],)),
                                                                group = np.tile(y_dict["training"][j],(n_sess[ds_name],)),
                                                                session = np.concatenate([np.repeat(j,len(y_dict["id"][j])) for j in range(n_sess[ds_name])]),
                                                                analyis_reg_vals = reg_models,
                                                                analyis_reg_names = analysis,
                                                                correlation_rdm_vals = corr_rdm_vals,
                                                                correlation_rdm_names = corr_rdm_names,
                                                                centering=False,
                                                                group_centering=False)

                    subroireglat_df = subroireglat_df.assign(subid=subid, roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    subroicorlat_df = subroicorlat_df.assign(subid=subid, roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    reg_res_df.append(subroireglat_df)
                    corr_res_df.append(subroicorlat_df)
                    
                    subroirdmlat_df = subroirdmlat_df.join(modeldf).reset_index()
                    subroirdmlat_df = subroirdmlat_df.assign(subid=subid, roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    all_roirdm_df.append(subroirdmlat_df)

            roirdm_df = pd.concat(all_roirdm_df,axis=0)
            roirdm_df.to_csv(os.path.join(ROIRSA_output_path,f'roirdm_nocentering_{p}_{ds_name}_{vselect}.csv'))

roirsareg_df = pd.concat(reg_res_df,axis=0)
roirsareg_df.to_csv(os.path.join(ROIRSA_output_path,f'roirsareg_nocentering.csv'))
roirsacor_df = pd.concat(corr_res_df,axis=0)
roirsacor_df.to_csv(os.path.join(ROIRSA_output_path,f'roirsacor_nocentering.csv'))