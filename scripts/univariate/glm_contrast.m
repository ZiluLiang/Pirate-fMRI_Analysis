function glm_contrast(SPMmat_dir,contrast_names,contrast_weights,flag_replace)
% Build contrast in glm in spm
% usage:
% glm_contrast(SPMmat_dir,contrast_names,contrast_weights,flag_replace)
% INPUTS:
% - SPMmat_dir: directory to spmMAT file
% - contrast_names: names of the contrasts to be built
% - contrast_weights: weight vector (for T contrast) or  
%                     weight matrix (for F contrast) for the contrasts
% - flag_replace: whether or not to delete the existing contrasts.
% -----------------------------------------------------------------------    
% Author: Zilu Liang
    
    if nargin<4, flag_replace = true; end

    % check if model exist
    if ~contains(SPMmat_dir,"SPM.mat")
        parent_dir = SPMmat_dir;
        SPMmat_dir = spm_select('FPList',parent_dir,"SPM.mat");
    else
        [parent_dir,~,~] = fileparts(SPMmat_dir);
    end
    if ~exist(SPMmat_dir,'file')
        error("SPM.mat file do not exist, please specify model before running estimation")
    end
    
    % check if number of contrast names and vectors match
    if  numel(contrast_names)==numel(contrast_weights)
        n_contrasts = numel(contrast_names);
    else
        error('number of contrast names, and number of contrast vectors do not match!')
    end

    % check if length of weight vector matches number of regressors
    if ~all(cellfun(@(w) numel(load(SPMmat_dir,'SPM').SPM.xX.name)==size(w,2),contrast_weights))
        error('number of columns in weight vector do not match total number of regressors!')
    end

    % define spm matlabbatch
    for j = 1:n_contrasts
        if size(contrast_weights{j},1) == 1
            %t.contrast
            contrast.consess{j}.tcon.name    = contrast_names{j};
            contrast.consess{j}.tcon.weights = contrast_weights{j};
            contrast.consess{j}.tcon.sessrep = 'none';                
        elseif size(contrast_weights{j},1) > 1
                %f.contrast
                contrast.consess{j}.fcon.name    = contrast_names{j};
                contrast.consess{j}.fcon.weights = contrast_weights{j};
                contrast.consess{j}.fcon.sessrep = 'none';                
        end
    end
    contrast.spmmat = cellstr(SPMmat_dir);
    contrast.delete = 1*flag_replace;%1 delete current contrasts;0 do not delete

    matlabbatch{1}.spm.stats.con = contrast;
    save(fullfile(parent_dir,'contrast.mat'),'matlabbatch');

    % run spm matlabbatch
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end