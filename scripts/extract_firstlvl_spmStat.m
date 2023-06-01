function [rangeCon,sumResMS,rangeStat] = extract_firstlvl_spmStat(glm_name,glm_dir,masks)
    [directory,participants]  = get_pirate_defaults(false,'directory','participants');
    contrast_names = {get_glm_config(glm_name).contrasts.name};
    if nargin<2
        glm_dir    = fullfile(directory.fmri_data,glm_name);
    end
    if nargin<3
        masks = repmat({'all'},numel(contrast_names),1);
    end
    
    rangeStat  = cell(numel(participants.validids),numel(contrast_names));
    rangeCon   = cell(numel(participants.validids),numel(contrast_names));
    sumResMS  = nan(numel(participants.validids), 3);
    for isub = 1:numel(participants.validids)
        firstlevel_dir = fullfile(glm_dir,'first',participants.validids{isub});
        for j = 1:numel(contrast_names)
            try
                [~,con_img,stat_img] = find_contrast_idx(fullfile(firstlevel_dir,'SPM.mat'),contrast_names{j});
                voxelwise_Stat  = spm_summarise(fullfile(firstlevel_dir,stat_img),masks{j});
                voxelwise_Con   = spm_summarise(fullfile(firstlevel_dir,con_img),masks{j});
                rangeStat{isub,j} = [min(voxelwise_Stat),max(voxelwise_Stat)];
                rangeCon{isub,j} = [min(voxelwise_Con),max(voxelwise_Con)];
            catch
                rangeStat{isub,j} = [nan,nan];
                rangeCon{isub,j} = [nan,nan];
            end           
        end        
        voxelwise_ResMS = spm_summarise(fullfile(firstlevel_dir,'ResMS.nii'),'all');            
        sumResMS(isub,:) = [min(voxelwise_ResMS),...
                             max(voxelwise_ResMS),...
                             mean(voxelwise_ResMS,'all','omitnan')];
    end
    rangeStat = cell2table(rangeStat,'VariableNames',contrast_names);
    rangeCon  = cell2table(rangeCon,'VariableNames',contrast_names);
    sumResMS = array2table(sumResMS,'VariableNames',{'min','max','mean'});
end
