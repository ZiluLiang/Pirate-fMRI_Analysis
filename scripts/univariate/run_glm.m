function err_tracker = run_glm(glm_name,glm_dir,preproc_img_dir,subidlist,steps)
    [directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
    
    if nargin < 2, glm_dir  = fullfile(directory.fmri_data,glm_name); end
    if nargin < 3, preproc_img_dir = directory.unsmoothed; end
    if nargin < 4, subidlist = participants.validids; end
    if nargin < 5, steps = {'spec2est','contrast','second_level'}; end
    
    nsub = numel(subidlist);
    err_tracker    = cell(2,1);
    
    err_tracker_1 = cell(nsub,1);
    for isub = 1:nsub
        fprintf('%s: Running first-level GLM for %s %d/%d subject\n', glm_name, subidlist{isub}, isub, nsub)
        subimg_dir = fullfile(preproc_img_dir,subidlist{isub});
        glm_config = get_glm_config(glm_name);           
        
        try 
            output_dir     = fullfile(glm_dir,'first',subidlist{isub});
            
            if ismember('spec2est',steps)
                nii_files      = cellstr(spm_select('FPList',subimg_dir,sprintf([glm_config.filepattern,'.*.nii'],strrep(subidlist{isub},'sub','')))); 
                [~,m_files]    = setup_multiconditions(glm_name,subidlist{isub},fullfile(glm_dir,'beh',subidlist{isub}));
                nuisance_files = cellstr(spm_select('FPList',...
                                         subimg_dir,...
                                         sprintf([filepattern.preprocess.nuisance,glm_config.filepattern,'.*.txt'],strrep(subidlist{isub},'sub',''))...
                                  ));
                              
                specify_estimate_glm(nii_files,m_files,nuisance_files,output_dir);
                fprintf('%s: Completed first-level GLM for %s %d/%d subject\n',glm_name, subidlist{isub}, isub, nsub)
            end
            if ismember('contrast',steps)
                specify_estimate_contrast(output_dir,...
                                          {glm_config.contrasts.name},...
                                          {glm_config.contrasts.type},...
                                          {glm_config.contrasts.wvec});
                fprintf('%s: Completed first-level contrast for %s %d/%d subject\n',glm_name,subidlist{isub}, isub, nsub)
            end
        catch err
            err_tracker_1{isub} = err;
        end
    end
    err_tracker{1} = err_tracker_1;
    
    if ismember('second_level',steps)
        try
            fprintf('%s: Running second-level\n', glm_name)
            specify_estimate_grouplevel(glm_dir,{glm_config.contrasts.name})
        catch err
            err_tracker{2} = err;
        end
    end
end