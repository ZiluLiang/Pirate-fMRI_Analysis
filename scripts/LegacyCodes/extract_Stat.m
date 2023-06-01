%clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
glm_names = {'sc_navigation','sc_localizer'};

rangeCon = struct();
meanResMS = struct();
rangeStat = struct();
for j = 1:numel(glm_names)
    glm_name = glm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
    rangeStat.(glm_name).Properties.RowNames = participants.validids;
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

function [contrast_idx,contrast_img,stat_img] = find_contrast_idx(subSPM,contrasts_names)
% INPUT:
%    - subSPM: path to first level SPM.mat file;
%    - contrasts_names: name or a cell array of names of the contrast(s);
% OUTPUT:
%    - contrast_idx: the index/indices of the contrast(s) in SPM.xCon;
%    - contrast_img: the name of the corresponding con***.nii file(s);
%    - stat_img: the name of the corresponding spmT***.nii or spmF***.nii
%    file(s);
    if ischar(subSPM)
        if exist(subSPM,'file')
            subSPM = load(subSPM).SPM;
        else
            error('SPM file do not exists')
        end
    elseif ~isstruct(subSPM)
        error('first input must be full path to SPM.mat file or the loaded SPM struct')
    end
    if ischar(contrasts_names)
        contrasts_names = cellstr(contrasts_names);
    elseif ~iscell(contrasts_names)
        error('second input must be a contrast name or a cell array of contrast names')
    end
        
    contrast_idx  = cellfun(@(cname) find(contains({subSPM.xCon.name},cname)),contrasts_names,'uni',0);
    contrast_idx(cellfun(@isempty,contrast_idx)) = {nan};
    contrast_idx = cell2mat(contrast_idx);
    [contrast_img,stat_img] = deal(cell(size(contrast_idx)));
    for j = 1:numel(contrast_idx)
        contrast_img{j} = sprintf('con_%04d.nii',contrast_idx);
        stat_img{j} = sprintf('spm%s_%04d.nii',subSPM.xCon(contrast_idx(j)).STAT,contrast_idx(j));
    end
    if numel(numel(contrast_idx)) == 1
        contrast_idx = contrast_idx(1);
        contrast_img = contrast_img{1};
        stat_img     = stat_img{1};
    end
end