function err_tracker = glm_runner(glm_name,steps,glm_dir,preproc_img_dir,subidlist,groupglm_pref,groupglm_cov)
% run first and second level analysis for different glms
% INPUT:
% - glm_name: name of the glm, this will be used to find glm
%             configurations in the glm_configure functions
% - steps: what steps to include, must be a struct with fields 'first' and
%          'second'. Field values specify steps to run in first/second level.
%   e.g. steps = struct('first', {{'specify','concatenate','estimate','contrast','result'}}, ...
%                       'second',{{'specify','estimate','contrast','result'}});
% - glm_dir: output directory for the glm model results
% - preproc_img_dir: directory for preprocessed fmri data
% - subidlist: list of participants to be included in the analysis
% - groupglm_pref: prefix added to second level (group level) glm names
% - groupglm_cov: covariate for group level glm. should be like:
%                   struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {}) or
%                   struct('files', {}, 'iCFI', {}, 'iCC', {})
% OUTPUT:
% - error_tracker: this function do not pause when spm job runs into error,
%                  errors are returned for tracking which participants' 
%                  data did not run through
%
% The following functions are called at different stage:
% glm_handles = struct('first_level_spec',  @glm_firstlevel, ...
%                      'concatenate',       @glm_concatenate, ...
%                      'estimate',          @glm_estimate, ...
%                      'contrast',          @glm_contrast, ...
%                      'result',            @glm_results,...
%                      'second_level_spec', @glm_grouplevel);
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO: make second level accommodate different types of first level
% contrasts

    spm('defaults','FMRI')
    [directory,participants]  = get_pirate_defaults(false,'directory','participants');
    
    def_steps = struct('first', {{'specify','estimate','contrast'}}, ...
                       'second',{{'specify','estimate','contrast','result'}});
    if nargin < 2 || isempty(steps),           steps = def_steps; end
    if nargin < 3 || isempty(glm_dir),         glm_dir  = fullfile(directory.fmri_data,glm_name); end
    if nargin < 4 || isempty(preproc_img_dir), preproc_img_dir = directory.smoothed; end
    if nargin < 5 || isempty(subidlist),       subidlist = participants.validids; end
    if nargin < 6 || isempty(groupglm_pref),   groupglm_pref = ''; end
    if nargin < 7 || isempty(groupglm_cov),    groupglm_cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {}); end


    nsub = numel(subidlist);
    err_tracker = cell(2,1);
    
    %% run first-level analysis
    err_tracker_1st = cell(nsub,1);
    if isfield(steps,'first') && ~isempty(steps.first)
        %parfor isub = 1:nsub
        for isub = 1:nsub
            err_tracker_1st(isub) = run_firstlevel(subidlist{isub},glm_name,glm_dir,preproc_img_dir,steps.first);
        end
        err_tracker{1} = err_tracker_1st;
    end

    %% run second-level analysis
    if isfield(steps,'second') && ~isempty(steps.second)
        err_tracker(2) = run_secondlevel(glm_name,glm_dir,steps.second,subidlist,groupglm_pref,groupglm_cov);
    end
end

function error_tracker = run_firstlevel(subid,glm_name,glm_dir,preproc_img_dir,steps)
    filepattern = get_pirate_defaults(false,'filepattern');
    subimg_dir  = fullfile(preproc_img_dir,subid);
    glm_config  = glm_configure(glm_name);    
    output_dir  = fullfile(glm_dir,'first',subid);
    error_tracker = {''};
    try
        if ismember('specify',steps)
            fprintf('%s: Specifying first-level GLM for %s \n', glm_name, subid)
            flag_concatenate = ismember('concatenate',steps);
            scans     = cellfun(@(x) numel(spm_vol(x)), ... % count how many volumes each 4D nii file has
                                cellstr(spm_select('FPList',subimg_dir,[glm_config.filepattern,'.*.nii'])))'; % get the 4D nii files
            [~,c_files,r_files] = setup_multi(glm_name,subid,fullfile(glm_dir,'beh',subid),glm_config.modelopt,flag_concatenate,preproc_img_dir);
            if flag_concatenate
                nii_files = {cellstr(spm_select('ExtFPList',subimg_dir,[glm_config.filepattern,'.*.nii']))};
            else
                nii_files = cellstr(spm_select('List',subimg_dir,[glm_config.filepattern,'.*.nii']));
                nii_files = cellfun(@(x) cellstr(spm_select('ExtFPList',subimg_dir,x)),nii_files,'UniformOutput',false);
            end            
            glm_firstlevel(nii_files,c_files,r_files,output_dir);
            
            if flag_concatenate
                glm_concatenate(output_dir,scans);
            end
        end

        if ismember('estimate',steps), glm_estimate(output_dir);  end
        
        if ismember('contrast',steps)        
            if ~isempty(glm_config.contrasts)
                fprintf('%s: Computing first-level contrast for %s \n', glm_name, subid)
                contrast_weights = cellfun(@(v) gen_contrast_matrix(fullfile(output_dir,'SPM.mat'),v),{glm_config.contrasts.wvec},'uni',0);
    
                glm_contrast(output_dir,...
                             {glm_config.contrasts.name},...
                             contrast_weights);
                fprintf('%s: Completed first-level contrast for %s \n',glm_name,subid)
            end
        end
    
        if ismember('result',steps),  glm_results(output_dir); end
        fprintf('%s: Completed first-level GLM for %s\n',glm_name, subid)
    catch err
        error_tracker{1} = err;
        fprintf('%s: failed first-level GLM for %s\n',glm_name, subid)
    end
end

function error_tracker = run_secondlevel(glm_name,glm_dir,steps,subidlist,groupglm_pref,groupglm_cov)
    error_tracker = {''};
    nsub = numel(subidlist);
    try
        fprintf('%s: Running second-level\n', glm_name)
        firstlvlSPM_dirs = fullfile(glm_dir,'first',subidlist);
        glm_config = glm_configure(glm_name);  
        for j = 1:numel(glm_config.contrasts)
            scans = cell(1,1);
            for isub = 1:nsub
                [~,contrast_img,~] = find_contrast_idx(fullfile(firstlvlSPM_dirs{isub},'SPM.mat'),glm_config.contrasts(j).name);
                scans{1,1}{isub} = fullfile(firstlvlSPM_dirs{isub},contrast_img);
            end
            second_level_dir = fullfile(glm_dir,'second',[groupglm_pref,glm_config.contrasts(j).name]);
            if ismember('specify',steps)
                glm_grouplevel(second_level_dir, ...
                               scans, ...
                               {glm_config.contrasts(j).name}, ...
                               groupglm_cov)
            end
            if ismember('estimate',steps), glm_estimate(second_level_dir);  end
            if ismember('contrast',steps)
                ncov = numel(groupglm_cov);
                groupcon_wv   = num2cell(eye(ncov+1),2);
                groupcon_name = [{glm_config.contrasts(j).name},{groupglm_cov.cname}];                    
                glm_contrast(second_level_dir,...
                             groupcon_name,...
                             groupcon_wv);
            end
            if ismember('result',steps), glm_results(second_level_dir); end
        end
        
        fprintf('%s: Completed second-level\n', glm_name)
    catch err
        error_tracker{1} = err;
        fprintf('%s: Failed second-level\n', glm_name)
    end
end