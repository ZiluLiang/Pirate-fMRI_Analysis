function [rangeCon,meanCon,meanResMS,rangeStat,meanStat] = extract_firstlvl_spmStat(glm_name,glm_dir,masks)
% This function finds the gives some summary statistics on the statistical
% results of first level glm analysis for sanity check of model fitting
% results.
% INPUT: 
%      glm_name: string, to get glm configuration using glm_configure
%      glm_dir:  string, directory to search from
%      masks:    cell array with number of elements equal to the number of
%      contrast in the glm configuration. Each element specify a Region of 
%      Interest(ROI) mask from which the stat values of the corresponding contrast are extracted. 
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO: calculate measure of model fitting quality?

    [directory,participants]  = get_pirate_defaults(false,'directory','participants');
    try
        contrast_names = {glm_configure(glm_name).contrasts.name};
    catch
        contrast_names = {};
    end
    if nargin<2 || isempty(glm_dir)
        glm_dir    = fullfile(directory.fmri_data,glm_name);
    end
    if nargin<3 || isempty(masks)
        masks = repmat({'all'},numel(contrast_names),1);
    end
    unique_masks = unique(masks);
    mask_names = cell(size(unique_masks));

    rangeStat  = cell(numel(participants.validids),numel(contrast_names));
    meanStat  = cell(numel(participants.validids),numel(contrast_names));    
    rangeCon   = cell(numel(participants.validids),numel(contrast_names));
    meanCon   = cell(numel(participants.validids),numel(contrast_names));
    meanResMS  = nan(numel(participants.validids), numel(unique_masks));
    
    for isub = 1:numel(participants.validids)
        firstlevel_dir = fullfile(glm_dir,'first',participants.validids{isub});
        for j = 1:numel(contrast_names)
            try
                [~,con_img,stat_img] = find_contrast_idx(fullfile(firstlevel_dir,'SPM.mat'),contrast_names{j});
                voxelwise_Stat  = spm_summarise(fullfile(firstlevel_dir,stat_img),masks{j});
                rangeStat{isub,j} = [min(voxelwise_Stat),max(voxelwise_Stat)];
                meanStat{isub,j} = mean(voxelwise_Stat);
                if ~isempty(con_img)
                    voxelwise_Con   = spm_summarise(fullfile(firstlevel_dir,con_img),masks{j});
                    rangeCon{isub,j} = [min(voxelwise_Con),max(voxelwise_Con)];
                    meanCon{isub,j} = mean(voxelwise_Con);
                else
                    rangeCon{isub,j} = [nan,nan];
                    meanCon{isub,j} = [nan,nan];
                end                
            catch
                rangeStat{isub,j} = [nan,nan];
                meanStat{isub,j} = [nan,nan];
                rangeCon{isub,j} = [nan,nan];
                meanCon{isub,j} = [nan,nan];
            end           
        end
        for k = 1:numel(unique_masks)
            voxelwise_ResMS = spm_summarise(fullfile(firstlevel_dir,'ResMS.nii'),unique_masks{k});            
            meanResMS(isub,k) = mean(voxelwise_ResMS,'all','omitnan');
            [~,mask_names{k},~] = fileparts(unique_masks{k});
        end
    end
    rangeStat = cell2table(rangeStat,'VariableNames',contrast_names);
    rangeCon  = cell2table(rangeCon,'VariableNames',contrast_names);
    meanStat  = cell2table(meanStat,'VariableNames',contrast_names);
    meanCon  = cell2table(meanCon,'VariableNames',contrast_names);
    meanResMS = array2table(meanResMS,'VariableNames',mask_names);
end
