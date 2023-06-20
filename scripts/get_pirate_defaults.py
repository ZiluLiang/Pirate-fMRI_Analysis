import joblib
import numpy as np
import os
import matlab.engine
import pandas as pd

def get_pirate_defaults():
    project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'

    eng = matlab.engine.start_matlab()
    subid_list = eng.eval("get_pirate_defaults(false,'participants').validids")

    fmri_output_path = os.path.join(project_path,'data','fmri')
    stim_list_fn = os.path.join(project_path,'scripts','generic','stimlist.txt')
    stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0)
    stim_id = np.array(stim_list['stim_id'])
    training_filter = np.array(stim_list['training'])

    LSA_GLM_dir = os.path.join(fmri_output_path,'smoothed5mmLSA')

    def get_oddeven_img(subid):
        firstlvl_dir = os.path.join(LSA_GLM_dir,'LSA_stimuli_navigation','first',subid)
        contrast_imgo = []
        contrast_imge = []
        for sid in stim_id:
            # call find_contrast_idx function in matlab to find the index of the corresponding contrasts
            eng.evalc("[~,contrast_imgo,~] = find_contrast_idx(fullfile('%s','SPM.mat'),'stim%02d_odd')" % (firstlvl_dir,sid))
            eng.evalc("[~,contrast_imge,~] = find_contrast_idx(fullfile('%s','SPM.mat'),'stim%02d_even')" % (firstlvl_dir,sid))
            contrast_imgo.append(eng.eval("contrast_imgo")) 
            contrast_imge.append(eng.eval("contrast_imge"))
        contrast_fns   = np.concatenate((contrast_imgo,contrast_imge))
        contrast_paths = [os.path.join(firstlvl_dir,fn) for fn in contrast_fns]
        conditions     = np.concatenate((['stim'+str(s)+'_odd' for s in stim_id],['stim'+str(s)+'_even' for s in stim_id]))
        mask_path = os.path.join(firstlvl_dir,'mask.nii')
        return subid,contrast_paths,mask_path,conditions

    def get_contrast_img(subid):
        firstlvl_dir = os.path.join(LSA_GLM_dir,'LSA_stimuli_navigation','first',subid)
        stimcon_fn = []
        for sid in stim_id:
            # call find_regressor_idx function in matlab to find the index of the corresponding regressor
            eng.evalc("[~,contrast_img,~] = find_contrast_idx(fullfile('%s','SPM.mat'),{regexpPattern('^stim%02d$')})" % (firstlvl_dir,sid))
            stimcon_fn.append(eng.eval("contrast_img")) 
        #eng.quit()
        conditions = ['stim%02d' % (s) for s in stim_id]
        mask_path = os.path.join(firstlvl_dir,'mask.nii')
        stimcon_paths = [os.path.join(firstlvl_dir,fn) for fn in stimcon_fn]
        return subid,stimcon_paths,mask_path,conditions

    def get_sessionreg_img(subid):
        firstlvl_dir = os.path.join(LSA_GLM_dir,'LSA_stimuli_navigation','first',subid)
        stimreg_fn = []
        for sid in stim_id:
            # call find_regressor_idx function in matlab to find the index of the corresponding regressor
            eng.evalc("[~,regimgnames] = find_regressor_idx(fullfile('%s','SPM.mat'),'stim%02d')" % (firstlvl_dir,sid))
            stimreg_fn.append(eng.eval("regimgnames")) 
        #eng.quit()
        conditions = ['stim%02d_run%d' % (s,r+1) for s in stim_id for r in range(4)]
        mask_path = os.path.join(firstlvl_dir,'mask.nii')
        stimreg_fn = np.concatenate(stimreg_fn, axis=0)
        stimreg_paths = [os.path.join(firstlvl_dir,fn) for fn in stimreg_fn]
        return subid,stimreg_paths,mask_path,conditions

    def get_localizereg_img(subid):
        localizerdir = os.path.join(LSA_GLM_dir,'LSA_stimuli_localizer')
        firstlvl_dir = os.path.join(localizerdir,'first',subid)
        stimreg_fn = []
        for sid in stim_id[np.where(training_filter==1)]:
            # call find_regressor_idx function in matlab to find the index of the corresponding regressor
            eng.evalc("[~,regimgnames] = find_regressor_idx(fullfile('%s','SPM.mat'),'stim%02d')" % (firstlvl_dir,sid))
            stimreg_fn.append(eng.eval("regimgnames")) 
        #eng.quit()
        conditions = ['stim%02d' % (s) for s in stim_id[np.where(training_filter==1)]]
        mask_path = os.path.join(firstlvl_dir,'mask.nii')
        stimreg_fn = np.concatenate(stimreg_fn, axis=0)
        stimreg_paths = [os.path.join(firstlvl_dir,fn) for fn in stimreg_fn]
        return subid,stimreg_paths,mask_path,conditions

    RSA_specs = dict()
    ## get odd and even separated
    print('retrieving participants contrast image directory\n')
    RSA_specs["oddeven"] = [get_oddeven_img(subid) for subid in subid_list]
    print('finished retrieving participants contrast image directory\n')

    ## get four separated sessions
    print('retrieving participants regressor image directory\n')
    RSA_specs["allsess"] = [get_sessionreg_img(subid) for subid in subid_list]
    print('finished retrieving participants regressor image directory\n')

    ## get stimuli activity averaged across four session
    print('retrieving participants contrast image directory\n')
    RSA_specs["stimave"] = [get_contrast_img(subid) for subid in subid_list]
    print('finished retrieving participants contrast image directory\n')

    ## get activity in localizer
    print('retrieving participants localizer task image directory\n')
    localizerRSAspecs = [get_localizereg_img(subid) for subid in subid_list]
    print('finished retrieving participants  localizer task image directory\n')
    eng.quit()
        
    joblib.dump(dict({'maintask':RSA_specs,'localizer':localizerRSAspecs}),
                os.path.join(LSA_GLM_dir,f'pattern_imgpath.pkl'))