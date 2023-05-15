function specify_estimate_contrast(varargin)
%writen by Zilu Liang (2023 May, Oxford)
%this script specifies contrast in first-level glm in spm
% usage: specify_estimate_contrast(firstlvl_output_dir,contrast_names,contrast_types,contrast_wvec,flag_replace)
    
    err_flag = 1;
    if nargin == 4 || nargin == 5
        if all(cellfun(@(arg) iscell(arg),varargin(2:4))) && ischar(varargin{1})
            firstlvl_output_dir = varargin{1};
            contrast_names      = varargin{2};
            contrast_types      = varargin{3};
            contrast_wvecs      = varargin{4};
            err_flag   = 0;
            if nargin<5, flag_replace = true; end
        end
    end
    
    if err_flag
        error('invalid inputs')
    end
    
    if numel(contrast_names)==numel(contrast_types) && numel(contrast_names)==numel(contrast_wvecs)
        n_contrasts = numel(contrast_names);
    else
        error('number of contrast names, contrast types, and number of contrast vectors do not match!')
    end   
    
    % define spm matlabbatch
    for j = 1:n_contrasts
        switch lower(contrast_types{j})
            case 't'   
                %t.contrast
                contrast.consess{j}.tcon.name    = contrast_names{j};
                contrast.consess{j}.tcon.weights = contrast_wvecs{j};
                contrast.consess{j}.tcon.sessrep = 'none';                
                
            case 'f'
                %f.contrast
                contrast.consess{j}.fcon.name    = contrast_names{j};
                contrast.consess{j}.fcon.weights = contrast_wvecs{j};
                contrast.consess{j}.fcon.sessrep = 'none';                
        end
    end
    contrast.spmmat = {fullfile(firstlvl_output_dir,'SPM.mat')};
    contrast.delete = 1*flag_replace;%1 delete current contrasts;0 do not delete

    matlabbatch{1}.spm.stats.con = contrast;
    save(fullfile(firstlvl_output_dir,'contrast_1stlvl.mat'),'matlabbatch');

    % run spm matlabbatch
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end