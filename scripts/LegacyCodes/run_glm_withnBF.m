function err_tracker = run_glm(glm_name,glm_dir,steps,preproc_img_dir,subidlist)
    spm('defaults','FMRI')
    [directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
    
    if nargin < 2, glm_dir  = fullfile(directory.fmri_data,glm_name); end
    if nargin < 3, steps = {'spec2est','contrast'}; end %'spec2est','contrast','second_level'}
    if nargin < 4, preproc_img_dir = directory.smoothed; end
    if nargin < 5, subidlist = participants.validids; end
    
    nsub = numel(subidlist);
    err_tracker = {cell(nsub,2),{}};
    
    %% run first-level analysis
    nN = 12;
    for isub = 1:nsub
        subimg_dir = fullfile(preproc_img_dir,subidlist{isub});
        glm_config = get_glm_config(glm_name);    
        output_dir = fullfile(glm_dir,'first',subidlist{isub});

        if ismember('spec2est',steps)
            try
                fprintf('%s: Estimating first-level GLM for %s %d/%d subject\n', glm_name, subidlist{isub}, isub, nsub)        
                nii_files      = cellstr(spm_select('FPList',subimg_dir,[glm_config.filepattern,'.*.nii'])); 
                [~,m_files]    = setup_multiconditions(glm_name,subidlist{isub},fullfile(glm_dir,'beh',subidlist{isub}),glm_config.modelopt);
                nuisance_files = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.nuisance,glm_config.filepattern,'.*.txt']));

                specify_estimate_glm(nii_files,m_files,nuisance_files,output_dir);
                fprintf('%s: Completed first-level GLM for %s %d/%d subject\n',glm_name, subidlist{isub}, isub, nsub)
            catch err
                err_tracker{1}{isub,1} = err;
            end
        end
        
        if ismember('contrast',steps)
            try
                fprintf('%s: Computing first-level contrast for %s %d/%d subject\n', glm_name, subidlist{isub}, isub, nsub)
                subSPM = load(fullfile(output_dir,'SPM.mat'),'SPM').SPM;
                contrast_weights = cellfun(@(v) gen_contrast_matrix(v,subSPM.xBF.order,nN,numel(subSPM.nscan)),{glm_config.contrasts.wvec},'uni',0);
                
                specify_estimate_contrast(output_dir,...
                                          {glm_config.contrasts.name},...
                                          contrast_weights);
                fprintf('%s: Completed first-level contrast for %s %d/%d subject\n',glm_name,subidlist{isub}, isub, nsub)
                report_results(output_dir)
            catch err
                err_tracker{1}{isub,2} = err;
            end
        end
        
    end
    
    %% run second-level analysis
    if ismember('second_level',steps)
        try
            fprintf('%s: Running second-level\n', glm_name)
            firstlvlSPM_dirs = fullfile(glm_dir,'first',subidlist);
            checkSPM = load(fullfile(firstlvlSPM_dirs{1},'SPM.mat'),'SPM').SPM;
            glm_config = get_glm_config(glm_name);  
            for j = 1:numel(glm_config.contrasts)
                % check number of basis function in 1 participant,
                % if multiple basis functions are used, run group level using one-way anova
                % otherwise use one-sample t test
                scans = cell(checkSPM.xBF.order,1);
                if checkSPM.xBF.order>1
                    for isub = 1:nsub
                        [contrast_idx,~] = construct_contrast_multibf(fullfile(firstlvlSPM_dirs{isub},'SPM.mat'),glm_config.contrasts(j).name,glm_config.contrasts(j).wvec);
                        for k = 1:checkSPM.xBF.order
                            scans{k,1}{isub} = fullfile(firstlvlSPM_dirs{isub},sprintf('con_%04d.nii',contrast_idx(k)));
                        end
                    end
                    specify_estimate_grouplevel(fullfile(glm_dir,'second',glm_config.contrasts(j).name),scans,{glm_config.contrasts(j).name})
                else
                    for isub = 1:nsub
                        [~,contrast_img,~] = find_contrast_idx(fullfile(firstlvlSPM_dirs{isub},'SPM.mat'),glm_config.contrasts(j).name);
                        scans{1,1}{isub} = fullfile(firstlvlSPM_dirs{isub},contrast_img);
                    end
                    specify_estimate_grouplevel(fullfile(glm_dir,'second',glm_config.contrasts(j).name),scans,{glm_config.contrasts(j).name})
                    specify_estimate_contrast(fullfile(glm_dir,'second',glm_config.contrasts(j).name),...
                                              {glm_config.contrasts(j).name},...
                                              {[1]});
                end                
                report_results(fullfile(glm_dir,'second',glm_config.contrasts(j).name))
            end
            
            fprintf('%s: Completed second-level\n', glm_name)
        catch err
            err_tracker{2} = err;
        end
    end
end

function [contrast_idx,contrasts_names_bf] = construct_contrast_multibf(subfirstlvldir,contrast_name,wvec)
    subSPM     = load(fullfile(subfirstlvldir,'SPM.mat'),'SPM').SPM;
    firstlvlF_wvec = gen_contrast_matrix(wvec,subSPM.xBF.order,nN,numel(subSPM.nscan));
    splitF2T_wvec  = arrayfun(@(k) firstlvlF_wvec(k,:),1:designSPM.xBF.order,'uni',0);
    contrasts_names_bf = arrayfun(@(k) sprintf('bf%d_%s',k,contrast_name),1:designSPM.xBF.order,'uni',0);
    
    % check if required contrasts already exists
    contrast_idx = cellfun(@(cname) find_contrast_idx(subSPM,cname),contrasts_names_bf);
    % build the required contrasts that do not exists yet.
    build_idx = arrayfun(@isnan,contrast_idx);
    if ~any(build_idx)
        specify_estimate_contrast(subfirstlvldir,...
                                  contrasts_names_bf(build_idx),...
                                  splitF2T_wvec(build_idx),...
                                  false); % do not replace existing contrast
    end
    % find contrast idx after all required contrasts have been constructed
    contrast_idx = find_contrast_idx(subSPM,contrasts_names_bf);
end
