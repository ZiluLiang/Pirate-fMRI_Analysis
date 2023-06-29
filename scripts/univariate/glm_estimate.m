function glm_estimate(SPMmat_dir,write_residuals)
% Estimate prespecified glm in spm
% usage: glm_estimate(SPMmat_dir,write_residuals)
% INPUT:
% - SPMmat_dir: directory to spmMAT file
% - write_residuals: whether or not to save residual files
%
% Author: Zilu Liang

    if ~contains(SPMmat_dir,"SPM.mat")
        parent_dir = SPMmat_dir;
        SPMmat_dir = spm_select('FPList',parent_dir,"SPM.mat");
    else
        [parent_dir,~,~] = fileparts(SPMmat_dir);
    end
    if ~exist(SPMmat_dir,'file')
        error("SPM.mat file do not exist, please specify model before running estimation")
    end

    if nargin<2, write_residuals=0;end

    %% Set up estimation batch job
    estimation.spmmat(1)        = cellstr(SPMmat_dir);
    estimation.write_residuals  = write_residuals;
    estimation.method.Classical = 1;  
    %% Save batchjob and run
    matlabbatch{1}.spm.stats.fmri_est = estimation;
    save(fullfile(parent_dir,'model_estimation.mat'),'matlabbatch')
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end