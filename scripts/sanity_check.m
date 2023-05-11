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
% ------ written by Zillu Liang(2023.4,Oxford)------

clear;clc
%% Configurations
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');

%% set up parallel pool
num_workers   = feature('NumCores');
poolobj       =  parpool(num_workers);%set up parallel processing
%create temporary variables so that we can minimize the amount of data sent to different parallel workers
ids           = participants.ids;
nsub          = participants.nsub;
                        
%% 
glm_name = 'sc_navigation';
parfor isub = 2:nsub-1
    fprintf('Running first-level GLM for %s %d/%d subject\n', ids{isub}, isub, nsub)
    specify_estimate_glm(glm_name,ids{isub})
    specify_estimate_contrast(glm_name,ids{isub})
    fprintf('Completed first-level GLM for %s %d/%d subject\n', ids{isub}, isub, nsub)
end

%% close parallel pool
delete(poolobj)