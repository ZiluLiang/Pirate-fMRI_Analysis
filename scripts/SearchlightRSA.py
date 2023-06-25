import numpy as np
import os
import json
import glob
import pandas as pd
import joblib
from multivariate.rsa_searchlight import RSASearchLight
from multivariate.rsa_estimator import MultipleRDMRegression, PatternCorrelation
from multivariate.helper import ModelRDM

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'
with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
    voxel_size = pirate_defaults['fmri']['voxelsize']

preprocess = ["unsmoothedLSA","smoothed5mmLSA"]
preprocess = ['unsmoothedLSA']
outputregexp = 'beta_%04d.nii'
outputcorrexp = 'rho_%04d.nii'
sphere_vox_count = dict(zip(preprocess,[[]]*len(preprocess)))
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
        pmask_flist.append(os.path.join(firstlvl_dir,'reliability_mask.nii'))

    analysis = {
        # image-based rdm
        "sc_betweens_stimuli":   ['between_stimuli'],
        "sc_alls_stimuli":       ['stimuli'],
        # feature-based: color(x) or shape(y)
        "sc_withins_feature1d":  ['within_feature1dx','within_feature1dy'],
        "sc_betweens_feature1d": ['between_feature1dx','between_feature1dy'],
        "sc_alls_feature1d":     ['feature1dx','feature1dy'],
        # # train-test
        # "tt_withins_stimuligroup":  ['within_stimuligroup'],
        # "tt_betweens_stimuligroup": ['between_stimuligroup'],
        # "tt_alls_stimuligroup":     ['stimuligroup'],
        # map-based
        "betweens_loc2d":  ["between_loc2d"],
        "withins_loc2d":   ["within_loc2d"],
        "alls_loc2d":      ["loc2d"],
        "betweens_loc1d":  ["between_loc1dx","between_loc1dy"],
        "withins_loc1d":   ["within_loc1dx","within_loc1dy"],
        "alls_loc1d":      ["loc1dx","loc1dy"],
        # map-based plus control
        "betweens_loc2d_c":  ["between_stimuligroup","between_loc2d"],
        "withins_loc2d_c":   ["within_stimuligroup","within_loc2d"],
        "alls_loc2d_c":      ["stimuligroup","loc2d"],
        "betweens_loc1d_c":  ["between_stimuligroup","between_loc1dx","between_loc1dy"],
        "withins_loc1d_c":   ["within_stimuligroup","within_loc1dx","within_loc1dy"],
        "alls_loc1d_c":      ["stimuligroup","loc1dx","loc1dy"]
        }

    beta_flist = {"fourruns":beta_flist4r,
                  "oddeven":beta_flistoe}
    n_sess = {"fourruns":4,
              "oddeven":2}
    mask_flist = {"noselection":fmask_flist,
                  "reliability_ths0":pmask_flist
                  }
    
    radius = voxel_size*4
    sphere_vox_count[p] = dict(zip(list(beta_flist.keys()),[[]]*len(beta_flist.keys())))
    for ds_name, ds in beta_flist.items():
        sphere_vox_count[p][ds_name] = dict(zip(list(mask_flist.keys()),[[]]*len(mask_flist.keys())))
        for vselect in mask_flist:
            sphere_vox_count[p][ds_name][vselect] = []
            for j,(subid,beta_path,m_path) in enumerate(zip(subid_list, ds, mask_flist[vselect])):
                print(f'{p} {vselect} : Running Searchlight RSA in {subid}')

                # instantiate RSA searchlight class
                subRSA = RSASearchLight(
                    patternimg_paths = beta_path,
                    mask_img_path=m_path,
                    process_mask_img_path=mask_flist["noselection"][j],
                    radius=radius,
                    njobs=20
                    )
                sphere_vox_count[p][ds_name][vselect].append(
                    np.array(subRSA.A.sum(axis=1)).squeeze()
                    )
                # build model rdm
                model_rdm = ModelRDM(stimid = y_dict["image"][j],
                                     stimloc = np.vstack([y_dict["locx"][j],y_dict["locy"][j]]).T,
                                     stimfeature = np.vstack([y_dict["color"][j],y_dict["shape"][j]]).T,# feature-based: color(x) or shape(y)
                                     stimgroup = y_dict["training"][j],
                                     n_session=n_sess[ds_name])

                # run search light
                for a_name,m_regs in analysis.items():
                    print(f'Analysis - {a_name}')
                    output_dir = os.path.join(LSA_GLM_dir,'searchlight_wb_rsa_regspr',vselect,ds_name,a_name,'first')
                    regress_models = [model_rdm.models[m] for m in m_regs]
                    subRSA.run(MultipleRDMRegression,regress_models,a_name,os.path.join(output_dir,subid), outputregexp, j == 0) # only show details at the first sub
                    subRSA.run(PatternCorrelation,regress_models,a_name,os.path.join(output_dir,subid), outputcorrexp, j == 0)

                print(f'{ds_name}: Completed searchlight in {subid}')
            joblib.dump(dict({'subid':subid_list,'count':sphere_vox_count[p][ds_name][vselect]}),
                        os.path.join(LSA_GLM_dir,'searchlight_wb_rsa_regspr',f'searchlight_voxcount_r4_{vselect}_{ds_name}.pkl'))

joblib.dump(sphere_vox_count,os.path.join(fmri_output_path,'searchlight_voxcount_r4.pkl'))