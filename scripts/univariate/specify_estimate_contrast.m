function specify_estimate_contrast(varargin)
%writen by Zilu Liang (2023 May, Oxford)
%this script specifies contrast in first-level glm in spm
% usage: specify_estimate_contrast(SPMmat_dir,contrast_names,contrast_weights,flag_replace)
    
    err_flag = 1;
    
    if all(cellfun(@(arg) iscell(arg),varargin(2:3))) && (ischar(varargin{1}) || isstring(varargin{1}))
        SPMmat_dir = varargin{1};
        contrast_names = varargin{2};
        contrast_weights = varargin{3};
        err_flag   = 0;            
    end
    if nargin<4
        flag_replace = true;
    else
        flag_replace = varargin{4};
    end

    
    if err_flag
        error('invalid inputs')
    end
    
    if  numel(contrast_names)==numel(contrast_weights)
        n_contrasts = numel(contrast_names);
    else
        error('number of contrast names, and number of contrast vectors do not match!')
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
    contrast.spmmat = {fullfile(SPMmat_dir,'SPM.mat')};
    contrast.delete = 1*flag_replace;%1 delete current contrasts;0 do not delete

    matlabbatch{1}.spm.stats.con = contrast;
    save(fullfile(SPMmat_dir,'contrast.mat'),'matlabbatch');

    % run spm matlabbatch
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end