function [contrast_idx,contrast_img,stat_img] = find_contrast_idx(subSPM,contrasts_names)
% This function finds the index of a contrast given its name in the SPM struct
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
    if ischar(contrasts_names) || isa(contrasts_names,'pattern')
        contrasts_names = {contrasts_names};
    elseif ~iscell(contrasts_names)
        error('second input must be a contrast name or a cell array of contrast names')
    end
        
    contrast_idx  = cellfun(@(cname) find(contains({subSPM.xCon.name},cname)),contrasts_names,'uni',0);
    contrast_idx(cellfun(@isempty,contrast_idx)) = {nan};
    contrast_idx = cell2mat(contrast_idx);
    [contrast_img,stat_img] = deal(cell(size(contrast_idx)));
    for j = 1:numel(contrast_idx)
        if ~isnan(contrast_idx(j))
            contrast_img{j} = sprintf('con_%04d.nii',contrast_idx(j));
            stat_img{j} = sprintf('spm%s_%04d.nii',subSPM.xCon(contrast_idx(j)).STAT,contrast_idx(j));
        end
    end
    if numel(numel(contrast_idx)) == 1
        contrast_idx = contrast_idx(1);
        contrast_img = contrast_img{1};
        stat_img     = stat_img{1};
    end
end