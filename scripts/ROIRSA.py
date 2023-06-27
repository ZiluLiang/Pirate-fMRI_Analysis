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
        "alls_loc2d_c":      ["stimuligroup",'feature1dx','feature1dy',"loc2d"]
        }
correlation_rdms = np.unique(sum(list(analysis.values()),[]))

anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'

anat_roi = ["hippocampus","parahippocampus","occipital","ofc"]
laterality = ["left","right","bilateral"]

def run_ROIRSA(beta_img, mask_imgs, regression_models, regressor_names,stimid,group,session,centering:bool=False,group_centering:bool=False):
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

    result = dict(zip(regression_models.keys(),[[]] * len(regression_models.keys())))
    for k,v in regression_models.items():
        MR = MultipleRDMRegression(neural_rdm,v)
        MR.fit()
        result[k] = MR.result
    df = []
    for k, v in regressor_names.items():
        reg_name_dict = dict(zip(range(len(v)),v))
        res_df = pd.DataFrame(result[k]).T.rename(columns = reg_name_dict).assign(analysis=k)
        df.append(res_df)
    return pd.concat(df,axis=0),neuralrdmdf


def modelrdm_to_df(rdm,stimid,group,session,modelname):
    lt,idx = lower_tri(rdm)
    c = np.repeat(np.arange(0, rdm.shape[0]), rdm.shape[0]).reshape(rdm.shape)[idx]
    i = np.tile(np.arange(0, rdm.shape[0]), rdm.shape[0]).reshape(rdm.shape)[idx]
    df = pd.DataFrame({'stimidA':stimid[c], 'stimidB': stimid[i], 
                       'groupA': group[c],  'groupB': group[i], 
                       'runA': session[c],    'runB': session[i], 
                        modelname: lt})
    return df

# preprocess - dataset - voxel selection - roi -laterality
all_df = []
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
                df_inc_models =['loc2d', 'loc1dx', 'loc1dy', 'feature2d', 'feature1dx', 'feature1dy', 'stimuli', 'stimuligroup']
                modeldf = [modelrdm_to_df(rdm       = model_rdm.models[m],
                                          stimid    = np.tile(y_dict["id"][j],(n_sess[ds_name],)),
                                          group     = np.tile(y_dict["training"][j],(n_sess[ds_name],)),
                                          session   = np.concatenate([np.repeat(j,len(y_dict["id"][j])) for j in range(n_sess[ds_name])]),
                                          modelname = m).set_index(
                                              ['stimidA', 'stimidB', 'groupA', 'groupB', 'runA', 'runB']
                                              ) for m in df_inc_models]
                modeldf = modeldf[0].join(modeldf[1:])

                reg_models = {k: [model_rdm.models[m] for m in v] for k, v in analysis.items()}
                for roi, lat in itertools.product(anat_roi, laterality):
                    print(f"{p} - {ds_name} - {vselect} - {subid} - {roi} = {lat}")
                    anat_mask = os.path.join(anatmaskdir,f'{roi}_{lat}.nii')
                    subroirsalat_df,subroirdmlat_df = run_ROIRSA(beta_img = ds[j],
                                                                 mask_imgs = [mask_flist[vselect][j],anat_mask], 
                                                                 regression_models = reg_models,
                                                                 regressor_names = analysis,
                                                                 stimid = np.tile(y_dict["id"][j],(n_sess[ds_name],)),
                                                                 group = np.tile(y_dict["training"][j],(n_sess[ds_name],)),
                                                                 session = np.concatenate([np.repeat(j,len(y_dict["id"][j])) for j in range(n_sess[ds_name])]),
                                                                 centering=False,
                                                                 group_centering=False)
                    
                    subroirsalat_df = subroirsalat_df.assign(subid=subid, roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    all_df.append(subroirsalat_df)
                    
                    subroirdmlat_df = subroirdmlat_df.join(modeldf).reset_index()
                    subroirdmlat_df = subroirdmlat_df.assign(subid=subid, roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    all_roirdm_df.append(subroirdmlat_df)

            ROIRSA_output_path = os.path.join(fmri_output_path,'ROIRSA',glm_name)
            checkdir(ROIRSA_output_path)
            roirdm_df = pd.concat(all_roirdm_df,axis=0)
            roirdm_df.to_csv(os.path.join(ROIRSA_output_path,f'roirdm_nocentering_{p}_{ds_name}_{vselect}.csv'))

roirsa_df = pd.concat(all_df,axis=0)
roirsa_df.to_csv(os.path.join(ROIRSA_output_path,f'roirsa_nocentering_{p}_{ds_name}_{vselect}.csv'))