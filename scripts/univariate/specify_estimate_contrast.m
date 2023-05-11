function specify_estimate_contrast(glm_name,subid,varargin)
    if nargin<3
        flag_replace = true;
    else
        flag_replace = varargin{1};
    end

    glm_config = get_glm_config(glm_name);
    directory  = get_pirate_defaults(false,'directory');
    firstlvl_output_dir = fullfile(directory.fmri_data,glm_config.name,'first',subid);
    
    % define spm matlabbatch
    for j = 1:numel(glm_config.contrasts)
        switch lower(glm_config.contrasts(j).type)
            case 't'   
                %t.contrast
                contrast.consess{j}.tcon.name    = glm_config.contrasts(j).name;
                contrast.consess{j}.tcon.weights = glm_config.contrasts(j).wvec;
                contrast.consess{j}.tcon.sessrep = 'none';                
                
            case 'f'
                %f.contrast
                contrast.consess{j}.fcon.name    = glm_config.contrasts(j).name;
                contrast.consess{j}.fcon.weights = glm_config.contrasts(j).wvec;
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
