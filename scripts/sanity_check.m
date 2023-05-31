clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
glm_names = {'sc_navigation','sc_localizer'};

flag_runGLM = false;
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
if flag_getStat
    rangeCon = struct();
    meanResMS = struct();
    rangeStat = struct();
    for j = 1:numel(glm_names)
        glm_name = glm_names{j};
        [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
        rangeStat.(glm_name).Properties.RowNames = participants.validids;
    end
end

function [rangeCon,meanResMS,rangeStat] = extract_firstlvl_spmStat(glm_name)
    [directory,participants]  = get_pirate_defaults(false,'directory','participants');
    masks = {fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','AAL_Occipital.nii'),...
             fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','HMAT_Motor.nii')};
    glm_dir    = fullfile(directory.fmri_data,glm_name);
    contrast_names = {get_glm_config(glm_name).contrasts.name};
    
    
    rangeStat  = cell(numel(participants.validids),numel(contrast_names));
    rangeCon   = cell(numel(participants.validids),numel(contrast_names));
    meanResMS  = nan(numel(participants.validids), numel(contrast_names));
    for isub = 1:numel(participants.validids)
        firstlevel_dir = fullfile(glm_dir,'first',participants.validids{isub});
        for j = 1:numel(contrast_names)
            [~,con_img,stat_img] = find_contrast_idx(fullfile(firstlevel_dir,'SPM.mat'),contrast_names{j});
            voxelwise_Stat  = spm_summarise(fullfile(firstlevel_dir,stat_img),masks{j});
            voxelwise_Con   = spm_summarise(fullfile(firstlevel_dir,con_img),masks{j});
            voxelwise_ResMS = spm_summarise(fullfile(firstlevel_dir,'ResMS.nii'),masks{j});            
            rangeStat{isub,j} = [min(voxelwise_Stat),max(voxelwise_Stat)];
            rangeCon{isub,j} = [min(voxelwise_Con),max(voxelwise_Con)];
            meanResMS(isub,j) = mean(voxelwise_ResMS,'all','omitnan');            
        end
    end
    rangeStat = cell2table(rangeStat,'VariableNames',contrast_names);
    rangeCon  = cell2table(rangeCon,'VariableNames',contrast_names);
    meanResMS = array2table(meanResMS,'VariableNames',contrast_names);
end

