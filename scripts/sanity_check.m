% sanity check GLMs of fmri data preprocessing
% after glm estimation:
% check visual and motor effect at group level
% check visual and motor effect at first level by extracting statistics
% from visual and motor ROI
% ------ written by Zilu Liang(2023.5,Oxford)------

clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
glm_names = {'sc_navigation','sc_localizer'};

flag_runGLM = true;
flag_getStat = true;

%% run sanity check GLMs
if flag_runGLM
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:numel(glm_names)
        glm_name = glm_names{j};
        err_tracker.(glm_name) = run_glm(glm_name);
    end
end

%% examine sanity check results - extract stats
masks = {fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','occipital_bilateral.nii'),...
         fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','HMAT_Motor.nii')};
if flag_getStat
    rangeCon = struct();
    meanResMS = struct();
    rangeStat = struct();
    for j = 1:numel(glm_names)
        glm_name = glm_names{j};
        [rangeCon.(glm_name),meanCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name),meanStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,glm_name),masks);
        rangeStat.(glm_name).Properties.RowNames = participants.validids;
    end
end