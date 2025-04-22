function [rangeCon,meanCon,meanResMS,rangeStat,meanStat] = extract_firstlvl_spmStat(glm_name,glm_dir,masks,contrast_names,subidlist)
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
    if nargin<2 || isempty(glm_dir)
        glm_dir    = fullfile(directory.fmri_data,glm_name);
    end
    if nargin<3 || isempty(masks)
        masks = repmat({'all'},numel(contrast_names),1);
    end
    if nargin<4 || isempty(contrast_names)
        try
            contrast_names = {glm_configure(glm_name).contrasts.name};
        catch
            contrast_names = {};
        end
    end
    if nargin<5 || isempty(subidlist)
        subidlist = participants.validids;
    end
    unique_masks = unique(masks);
    mask_names = cell(size(unique_masks));
    for k = 1:numel(unique_masks)
        [~,mask_names{k},~] = fileparts(unique_masks{k});
    end

    rangeStat  = cell(numel(subidlist), numel(unique_masks), numel(contrast_names));
    meanStat  = cell(numel(subidlist),  numel(unique_masks), numel(contrast_names));    
    rangeCon   = cell(numel(subidlist), numel(unique_masks), numel(contrast_names));
    meanCon   = cell(numel(subidlist),  numel(unique_masks), numel(contrast_names));
    meanResMS  = nan(numel(subidlist),  numel(unique_masks));
    
    for isub = 1:numel(subidlist)
        firstlevel_dir = fullfile(glm_dir,'first',subidlist{isub});
        fprintf('extracting %s\r',subidlist{isub})
        for j = 1:numel(contrast_names)
            [~,con_img,stat_img] = find_contrast_idx(fullfile(firstlevel_dir,'SPM.mat'),contrast_names{j});
            for k = 1:numel(unique_masks)
                if ~isempty(stat_img)
                    voxelwise_Stat  = spm_summarise(fullfile(firstlevel_dir,stat_img),unique_masks{k});
                    rangeStat{isub,k,j} = [min(voxelwise_Stat,[],'all','omitnan'),max(voxelwise_Stat,[],'all','omitnan')];
                    meanStat{isub,k,j} = mean(voxelwise_Stat,'all','omitnan');
                else
                    rangeStat{isub,k,j} = [nan,nan];
                    meanCon{isub,k,j} = [nan];
                end
                if ~isempty(con_img)
                    voxelwise_Con   = spm_summarise(fullfile(firstlevel_dir,con_img),unique_masks{k});
                    rangeCon{isub,k,j} = [min(voxelwise_Con,[],'all','omitnan'),max(voxelwise_Con,[],'all','omitnan')];
                    meanCon{isub,k,j} = mean(voxelwise_Con,'all','omitnan');
                else
                    rangeCon{isub,k,j} = [nan,nan];
                    meanCon{isub,k,j} = [nan];
                end        
            end
        
            voxelwise_ResMS = spm_summarise(fullfile(firstlevel_dir,'ResMS.nii'),unique_masks{k});            
            meanResMS(isub,k) = mean(voxelwise_ResMS,'all','omitnan');            
        end
    end
    rangeStat = cell2struct(arrayfun(@(x) cell2table(rangeStat(:,:,x),'VariableNames',mask_names),1:numel(contrast_names),UniformOutput=false)',contrast_names);
    rangeCon  = cell2struct(arrayfun(@(x) cell2table(rangeCon(:,:,x),'VariableNames',mask_names),1:numel(contrast_names),UniformOutput=false)',contrast_names);
    meanStat  = cell2struct(arrayfun(@(x) cell2table(meanStat(:,:,x),'VariableNames',mask_names),1:numel(contrast_names),UniformOutput=false)',contrast_names);
    meanCon   = cell2struct(arrayfun(@(x) cell2table(meanCon(:,:,x),'VariableNames',mask_names),1:numel(contrast_names),UniformOutput=false)',contrast_names);
    meanResMS = array2table(meanResMS,'VariableNames',mask_names);
end
