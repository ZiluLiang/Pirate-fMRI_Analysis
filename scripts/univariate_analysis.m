% This script runs the univariate analysis on repetition suppression and
% neural axis and train/test difference
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO need to rerun to test if new code is bug-free
clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
masks = cellstr(spm_select('FPList','D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat','.*.nii'));
%% run Repetition Supression GLMs
RSglm_names = {'rs_loc2d_navigation','rs_resploc2d_navigation','rs_loc2d_localizer','traintest_navigation'};
flag_runGLM  = true;
if flag_runGLM
    steps = struct('first', {{'specify','estimate','contrast'}}, ...
                   'second',{{'specify','estimate','contrast','result'}});
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:numel(RSglm_names)
        glm_name = RSglm_names{j};        
        err_tracker.(glm_name) = glm_runner(glm_name, glm_name);
        glm_runner(glm_name,rmfield(steps,'first'),'','',participants.learnerids,'learner_');
        glm_runner(glm_name,rmfield(steps,'first'),'','',participants.generalizerids,'generalizer_');
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(RSglm_names)
    glm_name = RSglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,'',masks);
end

%% run neural-axis analysis
NAglm_names = {'axis_loc_navigation','axis_resploc_navigation','axis_loc_localizer','axis_attrloc_navigation','axis_attryloc_navigation'};
flag_runGLM  = true;
if flag_runGLM
    steps = struct('first', {{'specify','estimate','contrast'}}, ...
                   'second',{{'specify','estimate','contrast','result'}});
    for j = numel(NAglm_names)
        glm_name = NAglm_names{j};
        err_tracker.(glm_name) = glm_runner(glm_name, steps);
        glm_runner(glm_name,rmfield(steps,'first'),'','',participants.learnerids,'learner_');
        glm_runner(glm_name,rmfield(steps,'first'),'','',participants.generalizerids,'generalizer_');
    end
end
%extract residuals to double check if models are running okay
for j = 1:numel(NAglm_names)
    glm_name = NAglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,'',masks);
end