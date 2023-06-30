function glm_concatenate(SPMmat_dir, scans)
%this script performs by-session adjustment for concatenated glm in spm
% usage:
% glm_concatenate(SPMmat_dir,contrast_names,contrast_weights,flag_replace)
% INPUTS:
% - SPMmat_dir: directory to spmMAT file
% - scans: number of scans in each session.
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    if ~contains(SPMmat_dir,"SPM.mat")
        SPMmat_dir = fullfile(SPMmat_dir,"SPM.mat");
    end
    if ~exist(SPMmat_dir,'file')
        error("SPM.mat file do not exist, please specify model before running concatenation")
    end
    spm_fmri_concatenate(char(SPMmat_dir),scans);
end