function err_tracker = run_glm(glm_name,steps,glm_dir,preproc_img_dir,subidlist)
% run first and second level analysis for different glms
% INPUT:
% - glm_name: name of the glm, this will be used to find glm
%             configurations in the get_glm_config functions
% - steps: what steps to include, must be from
%          {'spec2est','contrast','second_level'}. 
%          'spec2est': specify and estimat first level model
%          'contrast': specify and estimat first level contrast
%          'second_level': specify and estimat 2nd level model
% - glm_dir: output directory for the glm model results
% - preproc_img_dir: directory for preprocessed fmri data
% - subidlist: list of participants to be included in the analysis
% OUTPUT:
% - error_tracker: this function do not pause when spm job runs into error,
%                  errors are returned for tracking which participants' 
%                  data did not run through
% TODO: make second level accommodate different types of first level
% contrasts
    spm('defaults','FMRI')
    [directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
    
    if nargin < 2, steps = {'spec2est','contrast','second_level'}; end %'spec2est','contrast','second_level'}
    if nargin < 3, glm_dir  = fullfile(directory.fmri_data,glm_name); end
    if nargin < 4, preproc_img_dir = directory.smoothed; end
    if nargin < 5, subidlist = participants.validids; end
    
    nsub = numel(subidlist);
    err_tracker = {cell(nsub,2),{}};
    
    %% run first-level analysis
    for isub = 16:nsub
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
                if ~isempty(glm_config.contrasts)
                    fprintf('%s: Computing first-level contrast for %s %d/%d subject\n', glm_name, subidlist{isub}, isub, nsub)
                    contrast_weights = cellfun(@(v) gen_contrast_matrix(fullfile(output_dir,'SPM.mat'),v),{glm_config.contrasts.wvec},'uni',0);

                    specify_estimate_contrast(output_dir,...
                                              {glm_config.contrasts.name},...
                                              contrast_weights);
                    fprintf('%s: Completed first-level contrast for %s %d/%d subject\n',glm_name,subidlist{isub}, isub, nsub)
                    %report_results(output_dir)
                end
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
            glm_config = get_glm_config(glm_name);  
            for j = 1:numel(glm_config.contrasts)
                scans = cell(1,1);
                for isub = 1:nsub
                    [~,contrast_img,~] = find_contrast_idx(fullfile(firstlvlSPM_dirs{isub},'SPM.mat'),glm_config.contrasts(j).name);
                    scans{1,1}{isub} = fullfile(firstlvlSPM_dirs{isub},contrast_img);
                end
                specify_estimate_grouplevel(fullfile(glm_dir,'second',glm_config.contrasts(j).name),scans,{glm_config.contrasts(j).name})
                specify_estimate_contrast(fullfile(glm_dir,'second',glm_config.contrasts(j).name),...
                                          {glm_config.contrasts(j).name},...
                                          {[1]});
                report_results(fullfile(glm_dir,'second',glm_config.contrasts(j).name))
            end
            
            fprintf('%s: Completed second-level\n', glm_name)
        catch err
            err_tracker{2} = err;
        end
    end
end

% TODO: make second level accommodate different types of first level
% contrasts
% function [contrast_idx,split_names] = split_Fcontrast(subfirstlvldir,contrast_name,wvec,row_names)
% % construct_contrast_for_F
%    
%     subSPM     = load(fullfile(subfirstlvldir,'SPM.mat'),'SPM').SPM;
%     firstlvlF_wvec = gen_contrast_matrix(subSPM,wvec);
%     nC = size(firstlvlF_wvec,1);
%     splitF2T_wvec  = arrayfun(@(k) firstlvlF_wvec(k,:),1:nC,'uni',0);
%     split_names = arrayfun(@(k) sprintf('bf%d_%s',k,contrast_name),1:designSPM.xBF.order,'uni',0);
%     
%     % check if required contrasts already exists
%     contrast_idx = cellfun(@(cname) find_contrast_idx(subSPM,cname),split_names);
%     % build the required contrasts that do not exists yet.
%     build_idx = arrayfun(@isnan,contrast_idx);
%     if ~any(build_idx)
%         specify_estimate_contrast(subfirstlvldir,...
%                                   split_names(build_idx),...
%                                   splitF2T_wvec(build_idx),...
%                                   false); % do not replace existing contrast
%     end
%     % find contrast idx after all required contrasts have been constructed
%     contrast_idx = find_contrast_idx(subSPM,split_names);
% end
