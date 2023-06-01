% fmri data preprocessing: convert, rename to annonymized files and organize
% The script contains six steps controlled by six flags in the preprocess_flags struct:
%       calVDM:         calculated voxel displacement map
%       realign_unwarp: realign and unwarp using vdm
%       coregistration: coregister t1 to mean epi
%       segmentation:   segment and normalise coregistered T1 according to template
%       normalisation:  use the estimated normalization parameters to normalise epi images to mni space
%       smooth:         spatial smoothing
% Quality inspection should be taken in between steps to check preprocessing quality
%
% Dependencies:
%       - spm12, jsonlab-master
%       - renamer.json: the renamer.json consists of key-value pairs with new subject 
%                       names as keys, and old subject names as values, old subject names 
%                       are listed in an string array [annonymized id used in behavior file name, fmri file name]
%
% ------ written by Zilu Liang (2021.6,BNU)   ------
% ------ adapted by Zillu Liang(2023.4,Oxford)------

clear;clc
%% Configurations
% Specify directory
wk_dir          = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\';
script_dir      = fullfile(wk_dir,'scripts');
preprocess_dir  = fullfile(wk_dir,'data','fmri_image','preprocess');
renamer_fn      = fullfile(wk_dir,'data','renamer.json');
pm_default_file = fullfile(script_dir,'preprocessing','pm_defaults_Prisma_CIMCYC.m'); % files specifying fieldmap settings
    
SPM12_dir      = 'C:\Program Files\MATLAB\matlab toolbox\spm12';

% Configure the naming pattern of different type of scans, but field names of the filepattern struct must not be changed 
filepattern = struct('fieldmap',   struct('phasediff','^sub-.*_fmap-phasediff',...
                                          'shortecho','^sub-.*_fmap-magnitude1',...
                                          'longecho','^sub-.*_fmap-magnitude2'),...
                     'anatomical', struct('T1','^sub-.*_anat-T1w'),...
                     'functional', struct('task1','^sub-.*_task-piratenavigation_run-[1-4]',...% the first run in this will be used as the first session
                                          'task2','^sub-.*_task-localizer_run-[1]'));
                                     
% Configure the steps, field names of the preprocess_flags struct must not be changed 
preprocess_flags   = struct('calVDM',         false,...
                            'realign_unwarp', false,...
                            'coregistration', true,...
                            'segmentation',   false,...
                            'normalisation',  false,...
                            'smooth',         false);
                                     
                                     

                        
%% ------------------------------  Do not modify  ------------------------------
% functions called to perform preprocessing step 
preprocess_handles  = struct('calVDM',         @calculateVDM,...
                             'realign_unwarp', @realign_unwarp,...
                             'coregistration', @coregister,...
                             'segmentation',   @segment,...
                             'normalisation',  @normalise,...
                             'smooth',         @smooth);
                   
% prefix for files after preprocessing steps
preprocessed_prefix = struct('vdm','^vdm5_sc',...
                             'firstepiunwarp','^qc_ufirstepi_',...
                             'realignunwarp','^u',...
                             'meanepi','^mean',...
                             'motionparam','^rp_',...
                             'coreg','^r',...
                             'deformation','^y_',...
                             'normalise','^wu',... % normalise adds w, normalise is done on realigned unwarped images, so prefix is wu*
                             'smooth','^swu'); % smooth adds s, smooth is done on normalized realigned unwarped images, so prefix is swu*
filepattern.preprocess = preprocessed_prefix;

% Add path 
add_path(script_dir,1)
add_path(SPM12_dir,0)

% Read subject list 
renamer = loadjson(renamer_fn);
newids  = fieldnames(renamer);
nsub    = numel(newids);

%% -----------------------  Preprocess data  ---------------------- 
preprocess_steps = fieldnames(preprocess_handles);
preprocess_steps = preprocess_steps(cellfun(@(s) preprocess_flags.(s),preprocess_steps)); % only run steps where flag is true
poolobj =  parpool(3);
for j = 1:numel(preprocess_steps)
    curr_step   = preprocess_steps{j};
    curr_handle = preprocess_handles.(curr_step);
    fprintf('Running %s\n\n', curr_step)
    
    parfor isub = 1:nsub    %for isub = 1
        fprintf('Running %s %d/%d subject\n', newids{isub}, isub, nsub)
        subimg_dir  = fullfile(preprocess_dir,newids{isub});
        if curr_step == "calVDM"
            curr_handle(subimg_dir,filepattern,pm_default_file); %#ok<*PFBNS>
        else
            curr_handle(subimg_dir,filepattern);
        end
        fprintf('Completed %s %d/%d subject\n', newids{isub}, isub, nsub)
    end
    fprintf('Completed %s\n\n', curr_step)
end
delete(poolobj)

%% --------------  Quality Check in between steps  -------------- 
% n_tasks = 2;
% n_runs  = [4,1];
% n_volumes = [326] ;
% for isub = 1:nsub
%     subimg_dir  = fullfile(preprocess_dir,newids{isub});
%     for task = 1:n_tasks
%         QC('distortion_correction',subimg_dir,filepattern,task,1,1)
%     end
%     pause
% end
