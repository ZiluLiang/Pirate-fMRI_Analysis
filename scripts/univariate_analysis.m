clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
RSglm_names = {'rs_loc2d_navigation','rs_resploc2d_navigation','rs_loc2d_localizer','traintest_navigation'};
%% run Repetition Supression GLMs
flag_runGLM  = false;
if flag_runGLM
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:numel(RSglm_names)
        glm_name = v{j};
        err_tracker.(glm_name) = run_glm(glm_name,{'spec2est','contrast','second_level'});
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(RSglm_names)
    glm_name = RSglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
end


%% run LSA beta series extrator GLMs
LSAglm_names = {'LSA_stimuli_navigation','LSA_stimuli_localizer'};
flag_runGLM  = false;
if flag_runGLM
    for j = 1:numel(LSAglm_names)
    glm_name = LSAglm_names{j};
    err_tracker.(glm_name) = run_glm(glm_name,{'spec2est'},...
                                     fullfile(directory.fmri_data,'unsmoothedLSA',glm_name),...
                                     directory.unsmoothed);
    end
end

% extract residuals to double check if models are running okay
for j = 1:numel(LSAglm_names)
    glm_name = LSAglm_names{j};
    masks = cellstr(spm_select('FPList','D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\wfu','.*.nii'));
    [rangeCon.(glm_name),meanResMS1.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,'unsmoothedLSA',glm_name),masks);
    [rangeCon.(glm_name),meanResMS2.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,'smoothedLSA5mm',glm_name),masks);
end

