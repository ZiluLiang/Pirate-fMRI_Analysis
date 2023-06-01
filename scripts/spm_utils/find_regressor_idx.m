function regressor_idx = find_regressor_idx(subSPM,regressor_names)
% This function finds the index of regressors given its name/pattern in the SPM struct
% INPUT:
%    - subSPM: path to first level SPM.mat file;
%    - regressor_names: name or a cell array of names of the regressor(s);
% OUTPUT:
%    - regressor_idx: the index/indices of the regressor(s) in SPM.xX.name;
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