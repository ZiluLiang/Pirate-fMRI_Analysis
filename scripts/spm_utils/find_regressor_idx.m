function regressor_idx = find_regressor_idx(subSPM,regressor_names)
% This function finds the index of the name of regressors
% example usage: find_regressor_idx(subSPM,regressor_names)
% example usage: find_regressor_idx(subSPM,regressor_pattern)
%  if it is regular expression, it will return all the matches of that reg
%  exp.
%    subSPM = load(fullfile(glm_dir,'first',subid,'SPM.mat'),'SPM').SPM;
    if ischar(subSPM)
        if exist(subSPM,'file')
            subSPM = load(subSPM).SPM;
        else
            error('SPM file do not exists')
        end
    elseif ~isstruct(subSPM)
        error('first input must be full path to SPM.mat file or the loaded SPM struct')
    end
    if ischar(regressor_names)
        regressor_names = cellstr(regressor_names);
    elseif ~iscell(regressor_names)
        error('second input must be a regressor name/pattern or a cell array of regressor names/patterns')
    end
        
    regressor_idx  = cellfun(@(cname) find(contains(subSPM.xX.name,cname)),regressor_names,'uni',0);
    regressor_idx(cellfun(@isempty,regressor_idx)) = {nan};
    regressor_idx = cell2mat(regressor_idx);
end