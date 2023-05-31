clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
glm_names = {'rs_loc2d_navigation','rs_resploc2d_navigation','rs_loc2d_localizer',...
             'traintest_navigation','LSA_stimuli_navigation','LSA_stimuli_localizer'};%,'rs_loc2d_localizer'};
flag_runGLM  = true;
%% run GLMs
if flag_runGLM
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:3%numel(glm_names):-1:1
        glm_name = glm_names{j};
        err_tracker.(glm_name) = run_glm(glm_name,{'spec2est','contrast','second_level'});
    end
end
%% extract residuals to double check if models are running okay
for j = numel(glm_names):-1:1
    glm_name = glm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
end