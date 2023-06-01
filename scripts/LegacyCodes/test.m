function err_tracker = test(glm_name,glm_dir,steps,num_workers,preproc_img_dir,subidlist)
    spm('defaults','FMRI')
    [directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
    
    if nargin < 2, glm_dir  = fullfile(directory.fmri_data,glm_name); end
    if nargin < 3, steps = {'second_level'}; end % {'spec2est','contrast','second_level'}
    if nargin < 4, num_workers = feature('NumCores')-3; end    
    if nargin < 5, preproc_img_dir = directory.smoothed; end
    if nargin < 6, subidlist = participants.validids; end
    
    nsub = numel(subidlist);
    err_tracker    = cell(2,1);
    
    %% run first-level analysis
    if ismember('spec2est',steps) || ismember('contrast',steps)
        err_tracker_1 = cell(nsub,1);
        poolobj       =  parpool(num_workers);%#ok<*NASGU> %set up parallel processing
    
        parfor isub = 1:nsub
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
                    subSPM = load(fullfile(output_dir,'SPM.mat'),'SPM').SPM;
                    contrast_weights = cellfun(@(v) gen_contrast_matrix(v,subSPM.xBF.order,12,numel(subSPM.nscan)),{glm_config.contrasts.wvec},'uni',0);
                    specify_estimate_contrast(output_dir,...
                                              {glm_config.contrasts.name},...
                                              contrast_weights);
                    fprintf('%s: Completed first-level contrast for %s %d/%d subject\n',glm_name,subidlist{isub}, isub, nsub)
                    report_results(output_dir)
                end
            catch err
                err_tracker_1{isub} = err;
            end
        end
        err_tracker{1} = err_tracker_1;
        delete(poolobj)
    end
    
    %% run second-level analysis
    if ismember('second_level',steps)
        try
            fprintf('%s: Running second-level\n', glm_name)
            % check number of basis function in 1 participant, if greater
            % than one run group level using one-way anova, else use
            % one-sample t test
            nBF_check = load(fullfile(glm_dir,'first',subidlist{1},'SPM.mat'),'SPM').SPM.xBF.order;
            if nBF_check>1
                % generate first level contrast for the analysis
                for isub = 1:nsub
                    subimg_dir = fullfile(preproc_img_dir,subidlist{isub});
                    glm_config = get_glm_config(glm_name);
                    output_dir = fullfile(glm_dir,'first',subidlist{isub});
                    subSPM     = load(fullfile(output_dir,'SPM.mat'),'SPM').SPM;
                    firstlvlcontrast_weights = cellfun(@(v) gen_contrast_matrix(v,subSPM.xBF.order,12,numel(subSPM.nscan)),{glm_config.contrasts.wvec},'uni',0);
                    splitF2T_wvec   = cell(numel(glm_config.contrasts),1);
                    contrasts_names = cell(numel(glm_config.contrasts),1);
                    for j = 1:numel(firstlvlcontrast_weights)
                        splitF2T_wvec{j}   = arrayfun(@(x) firstlvlcontrast_weights{j}(k,:),1:nBF,'uni',0);
                        contrasts_names{j} = arrayfun(@(k) sprintf('bf%d_%s',k,glm_config.contrasts(j).name),1:nBF,'uni',0);
                    end
                    specify_estimate_contrast(output_dir,...
                                              cat(1,contrasts_names{:}),...
                                              cat(1,splitF2T_wvec{:}),...
                                              false); % do not replace existing contrast
                end
            end
            
            designSPM = load(fullfile(glm_dir,'first',subidlist{1},'SPM.mat'),'SPM').SPM;
            glm_config = get_glm_config(glm_name);  
            for j = 1:numel(glm_config.contrasts)
                model2_name = glm_config.contrasts(j).name;
                scans = cell(designSPM.xBF.order,1);
                for k = 1:designSPM.xBF.order
                    cname = sprintf('bf%d_%s',k,glm_config.contrasts(j).name);
                    cind  = find(contains(cname,{designSPM.xCon.name}));
                    scans{k,1} = fullfile(glm_dir,'first',subidlist,sprintf('con_%04d.nii',cind));
                end
                specify_estimate_grouplevel(fullfile(glm_dir,model2_name),scans,{model2_name})
            end
        catch err
            err_tracker{2} = err;
        end
    end
end