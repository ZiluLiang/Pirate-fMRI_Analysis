import itertools
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import ModelRDM, compute_rdm,checkdir,lower_tri,upper_tri
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
import pandas as pd
import glob
from sklearn.manifold import MDS
import numpy
import matplotlib

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'

with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    learnerid_list = pirate_defaults['participants']['learnerids']
    generalizerid_list = pirate_defaults['participants']['generalizerids']

preprocess = ["smoothed5mmLSA","unsmoothedLSA"]

anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'

anat_roi = ["hippocampus","parahippocampus","occipital","ofc"]
laterality = ["left","right","bilateral"]

def run_ROIMDS(beta_img, mask_imgs,subid, group=None):
    APD = ActivityPatternDataLoader(beta_img,mask_imgs)
    activitypattern = APD.X
    embedding = MDS(
        n_components=1,
        max_iter=50000,
        n_init=100,
        n_jobs=10,
        random_state=94,
        dissimilarity = "precomputed",
        normalized_stress=False,
    )
    pos = embedding.fit_transform(compute_rdm(activitypattern,'correlation'))
    newidx = []
    pos = [[]] * np.unique(group).size
    for k in np.unique(group):
        group_activitypattern = activitypattern[np.where(group == k)[0],:]
        newidx.append(np.where(group == k)[0])
        pos[k] = embedding.fit_transform(compute_rdm(group_activitypattern,'correlation'))
    pos_dfs = [pd.DataFrame({"subid":np.repeat(subid,p.shape[0]),"x":p[:,0],"y":p[:,1]}) for p in pos]
    return pd.concat(pos_dfs,axis=0),np.concatenate(newidx)

def run_groupROIMDS(beta_img_list, mask_imgs_list, group=None, ncomponents=2):    
    activitypattern_list = []
    for beta_img,mask_imgs in zip(beta_img_list,mask_imgs_list):
        APD = ActivityPatternDataLoader(beta_img,mask_imgs)
        activitypattern_list.append(APD.X)

    neuralrdm_list =  []
    for j,k in enumerate(np.unique(group)):
        neuralrdm_list.append([])
        for activitypattern in activitypattern_list:
            group_activitypattern = activitypattern[np.where(group == k)[0],:]            
            neuralrdm_list[j].append(compute_rdm(group_activitypattern,'correlation'))            

    embedding = MDS(
        n_components=ncomponents,
        max_iter=50000,
        n_init=100,
        n_jobs=10,
        random_state=94,
        dissimilarity = "precomputed",
        normalized_stress=False,
    )

    pos = [[]] * np.unique(group).size
    newidx = []
    for j,k in enumerate(np.unique(group)):
        newidx.append(np.where(group == k)[0])
        mean_grouprdm = np.mean(neuralrdm_list[j],axis=0)
        pos[j] = embedding.fit_transform(compute_rdm(mean_grouprdm,'correlation'))

    col_name_by_compo = {2:{0:"x",1:"y"},
                         3:{0:"x",1:"y",2:"z"},
                         4:{0:"x",1:"y",2:"u",3:"v"}}
    col_names = (
        col_name_by_compo[ncomponents]
        if ncomponents in col_name_by_compo
        else [f'x{str(c)}' for c in range(ncomponents)]
    )
    pos_dfs = [pd.DataFrame(p).rename(columns = col_names) for p in pos]
    return pd.concat(pos_dfs,axis=0),np.concatenate(newidx)

# preprocess - ds - vselect - roi -laterality
all_df = []
all_groupdf = []
#    for ds_name, ds in beta_flist.items():
ds_name = 'mu'
for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)
    ########################## Get Stimuli Data ##############################################
    beta_flist4r = []
    beta_flistoe = []
    beta_flistmu = []
    fmask_flist = []
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
        beta_flistmu.append(os.path.join(firstlvl_dir,'stimuli_mu.nii'))  
        fmask_flist.append(os.path.join(firstlvl_dir,'mask.nii'))
        pmask_flist.append(os.path.join(firstlvl_dir,'reliability_mask.nii'))

    ########################## Run RSA ##############################################
    n_sess = {"fourruns":4,
              "oddeven":2,
              "mu":1}
    beta_flist = {"fourruns":beta_flist4r,
                "oddeven":beta_flistoe}
    mask_flist = {"noselection":fmask_flist,
                "reliability_ths0":pmask_flist}
    learner_filter = [x in learnerid_list for x in subid_list]
    generalizer_filter = [x in generalizerid_list for x in subid_list]

    ds = beta_flistmu
    for vselect in mask_flist:            
        result = []       
        for roi, lat in itertools.product(anat_roi, laterality):
            print(f"{p} - {ds_name} - {vselect} - {roi} = {lat}")
            # for j,subid in enumerate(subid_list):            
            #     print(f"{p} - {ds_name} - {vselect} - {subid} - {roi} = {lat}")
            #     anat_mask = os.path.join(anatmaskdir,f'{roi}_{lat}.nii')
            #     subriolat_df,idx = run_ROIMDS(beta_img = ds[j],
            #                             mask_imgs = [mask_flist[vselect][j],anat_mask],
            #                             subid=subid,
            #                             group=numpy.tile(y_dict["training"][j],(n_sess[ds_name],)))
            #     subriolat_df = subriolat_df.assign(roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
            #     subriolat_df["locx"] = y_dict["locx"][j][idx]
            #     subriolat_df["locy"] = y_dict["locy"][j][idx]
            #     subriolat_df["color"] = y_dict["color"][j][idx]
            #     subriolat_df["shape"] = y_dict["shape"][j][idx]
            #     subriolat_df["training"] = y_dict["training"][j][idx]

            #     all_df.append(subriolat_df)
            anat_mask = os.path.join(anatmaskdir,f'{roi}_{lat}.nii')
            all_sub_mlist = [[m,anat_mask] for m in mask_flist[vselect]]
            for ncomponents in [2,3,4]:
                allsub_df,idxa  = run_groupROIMDS(beta_img_list  = ds,
                                             mask_imgs_list = all_sub_mlist,
                                             group          = numpy.tile(y_dict["training"][0],(n_sess[ds_name],)),
                                             ncomponents    = ncomponents)
                allsub_df = allsub_df.assign(
                                                subgroup = "all",
                                                ncomponents=ncomponents,
                                                roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p
                                             )
                learner_df,idxl = run_groupROIMDS(beta_img_list  = list(np.array(ds)[learner_filter]),
                                             mask_imgs_list = [ms for j,ms in enumerate(all_sub_mlist) if learner_filter[j]],
                                             group          = numpy.tile(y_dict["training"][0],(n_sess[ds_name],)),
                                             ncomponents    = ncomponents)
                learner_df = learner_df.assign(
                                                subgroup = "learner",
                                                ncomponents=ncomponents,
                                                roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p
                                             )
                generalizer_df,idxg = run_groupROIMDS(beta_img_list  = list(np.array(ds)[generalizer_filter]),
                                                 mask_imgs_list = [ms for j,ms in enumerate(all_sub_mlist) if generalizer_filter[j]],
                                                 group          = numpy.tile(y_dict["training"][0],(n_sess[ds_name],)),
                                                 ncomponents    = ncomponents)
                generalizer_df = generalizer_df.assign(
                                                subgroup = "generalizer",
                                                ncomponents=ncomponents,
                                                roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p
                                             )
                tmpdf_list = []
                for tmpdf,idx in zip([allsub_df,learner_df,generalizer_df],[idxa,idxl,idxg]):
                    tmpdf["locx"] = y_dict["locx"][0][idx]
                    tmpdf["locy"] = y_dict["locy"][0][idx]
                    tmpdf["color"] = y_dict["color"][0][idx]
                    tmpdf["shape"] = y_dict["shape"][0][idx]
                    tmpdf["training"] = y_dict["training"][0][idx]
                    tmpdf_list.append(tmpdf)
                all_groupdf.append(pd.concat(tmpdf_list,axis=0))

ROIRSA_output_path = os.path.join(fmri_output_path,'ROIRSA',glm_name+'_mean')
checkdir(ROIRSA_output_path)
#df = pd.concat(all_df,axis=0)
#df.to_csv(os.path.join(ROIRSA_output_path,'roimds.csv'))

groupdf = pd.concat(all_groupdf,axis=0)
groupdf.to_csv(os.path.join(ROIRSA_output_path,'grouproimds.csv'))
# id_cols = ["ds","preprocess","voxselect","roi","laterality",'subid','analysis']
# for k,v in analysis.items():
#     plot_df = df.loc[df["analysis"]==k,tuple(id_cols+v)]
#     plot_df["roi_l"] = plot_df[['roi', 'laterality']].apply(lambda x: '_'.join(x), axis=1)
#     plot_df["ds_preproc"] = plot_df[['ds', 'preprocess']].apply(lambda x: '_'.join(x), axis=1)
#     plot_df = pd.melt(plot_df, id_vars=id_cols+["roi_l","ds_preproc"], value_vars=v)
#     g = sns.catplot(data=plot_df, x="roi_l", y="value",
#     hue="voxselect",col="variable",row="ds_preproc",
#     kind="box", aspect=3,sharex=True,sharey=True)
#     tmp = [plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
#     g.savefig(os.path.join(ROIRSA_output_path,f'{k}.png'))