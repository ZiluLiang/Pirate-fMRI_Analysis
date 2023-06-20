import itertools
import os
import numpy as np
import nibabel as nib
import nibabel.processing
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import json
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.masking import apply_mask,intersect_masks

from nilearn.image import new_img_like
from nilearn.decoding import SearchLight
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import compute_rdm
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
import joblib


import pandas as pd
import glob

from multivariate.helper import ModelRDM

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'

preprocess = ["smoothed5mmLSA","unsmoothedLSA"]
with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
n_run = 4

for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)
    beta_flist4r = []
    beta_flistoe = []
    fmask_flist = []
    pmask_flist = []
    run_stim_labels = []
    y_dict = {"id":[],
            "image":[],
            "locx":[],
            "locy":[],
            "color":[],
            "shape":[]} 

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

        # build list of beta maps
        firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)

        y_dict["id"].append(stim_id)
        y_dict["image"].append(stim_image)
        y_dict["locx"].append(stim_locx)
        y_dict["locy"].append(stim_locy)
        y_dict["color"].append(stim_color)
        y_dict["shape"].append(stim_shape)

        beta_flist4r.append(os.path.join(firstlvl_dir,'stimuli_4r.nii'))    
        beta_flistoe.append(os.path.join(firstlvl_dir,'stimuli_oe.nii'))    
        fmask_flist.append(os.path.join(firstlvl_dir,'mask.nii'))
        pmask_flist.append(os.path.join(firstlvl_dir,'reliability_mask.nii'))

analysis = {
    # image-based rdm
    "sc_betweens_stimuli":   ['between_stimuli'],
    "sc_alls_stimuli":       ['stimuli'],
    # feature-based: color(x) or shape(y)
    "sc_withins_feature1d":  ['within_feature1dx','within_feature1dy'],
    "sc_betweens_feature1d": ['between_feature1dx','between_feature1dy'],
    "sc_alls_feature1d":     ['feature1dx','feature1dy'],
    # map-based
    "betweens_loc2d":  ["between_loc2d"],
    "withins_loc2d":   ["within_loc2d"],
    "alls_loc2d":      ["loc2d"],
    "betweens_loc1d":  ["between_loc1dx","between_loc1dy"],
    "withins_loc1d":   ["within_loc1dx","within_loc1dy"],
    "alls_loc1d":      ["loc1dx","loc1dy"]
    }
anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'

anat_roi = ["hippocampus","parahippocampus","occipital","ofc"]
laterality = ["left","right","bilateral"]

beta_flist = {"fourruns":beta_flist4r,
              "oddeven":beta_flistoe}
n_sess = {"fourruns":4,
          "oddeven":2}
mask_flist = {"noselection":fmask_flist,
              "reliability_ths0":pmask_flist}

def run_ROIRSA(beta_img, mask_imgs, regression_models, regressor_names,subid):
    APD = ActivityPatternDataLoader(beta_img,mask_imgs)
    result = dict(zip(regression_models.keys(),[[]] * len(regression_models.keys())))
    for k,v in regression_models.items():
        MR = MultipleRDMRegression(compute_rdm(APD.X,'correlation'),v)
        MR.fit()
        result[k] = MR.result
    df = []
    for k, v in regressor_names.items():
        reg_name_dict = dict(zip(range(len(v)),v))
        res_df = pd.DataFrame(result[k]).T.rename(columns = reg_name_dict).assign(analysis=k,subid=subid)
        df.append(res_df)
    return pd.concat(df,axis=0)

# preprocess - ds - vselect - roi -laterality
all_df = []
for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)
    for ds_name, ds in beta_flist.items():
        for vselect in mask_flist:            
            result = dict(zip(analysis.keys(),[[]] * len(analysis.keys())))
            for j,subid in enumerate(subid_list):
                for roi, lat in itertools.product(anat_roi, laterality):
                    print(f"{p} - {ds_name} - {vselect} - {subid} - {roi} = {lat}")
                    anat_mask = os.path.join(anatmaskdir,f'{roi}_{lat}.nii')

                    model_rdm = ModelRDM(y_dict["image"][j],
                                        np.vstack([y_dict["locx"][j],y_dict["locy"][j]]).T,
                                        np.vstack([y_dict["color"][j],y_dict["shape"][j]]).T,
                                        n_session=n_sess[ds_name])
                    reg_models = {k: [model_rdm.models[m] for m in v] for k, v in analysis.items()}
                    subriolat_df = run_ROIRSA(beta_img = ds[j],
                                              mask_imgs = [mask_flist[vselect][j],anat_mask],
                                              regression_models = reg_models,
                                              regressor_names=analysis,
                                              subid=subid)
                    subriolat_df = subriolat_df.assign(roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    all_df.append(subriolat_df)
df = pd.concat(all_df,axis=0)
df.to_csv(os.path.join(fmri_output_path,'roirsa.csv'))

id_cols = ["ds","preprocess","voxselect","roi","laterality",'subid','analysis']
for k,v in analysis.items():
    plot_df = df.loc[df["analysis"]==k,tuple(id_cols+v)]
    plot_df["roi_l"] = plot_df[['roi', 'laterality']].apply(lambda x: '_'.join(x), axis=1)
    plot_df["ds_preproc"] = plot_df[['ds', 'preprocess']].apply(lambda x: '_'.join(x), axis=1)
    plot_df = pd.melt(plot_df, id_vars=id_cols+["roi_l","ds_preproc"], value_vars=v)
    g = sns.catplot(data=plot_df, x="roi_l", y="value",
    hue="voxselect",col="variable",row="ds_preproc",
    kind="box", aspect=3,sharex=True,sharey=True)
    tmp = [plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
    g.savefig(os.path.join(fmri_output_path,f'{k}.png'))