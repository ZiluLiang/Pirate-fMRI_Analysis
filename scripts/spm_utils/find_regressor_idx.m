function [regressor_idx,regressor_img] = find_regressor_idx(subSPM,regname_patterns)
% This function finds the index of regressors given its name/pattern in the SPM struct
% INPUT:
%    - subSPM: path to first level SPM.mat file;
%    - regname_patterns: pattern string or a cell array of pattern strings
%                        of the regressor(s) name; see MATLAB documentation
%                        of `contain` and `pattern` for details.
% OUTPUT:
%    - regressor_idx: the index/indices of the regressor(s) in SPM.xX.name;
%    - regressor_img: the name of the corresponding beta image;
%
% Author: Zilu Liang

    if ischar(subSPM)
        if exist(subSPM,'file')
            subSPM = load(subSPM).SPM;
        else
            error('SPM file do not exists')
        end
    elseif ~isstruct(subSPM)
        error('first input must be full path to SPM.mat file or the loaded SPM struct')
    end
    if ischar(regname_patterns) || isa(regname_patterns,'pattern')
        regname_patterns = {regname_patterns};
    elseif ~iscell(regname_patterns)
        error('second input must be a regressor name/pattern or a cell array of regressor names/patterns')
    end
        
    regressor_idx  = cellfun(@(cname) find(contains(subSPM.xX.name,cname)),regname_patterns,'uni',0);
    regressor_idx(cellfun(@isempty,regressor_idx)) = {nan};
    regressor_idx = cell2mat(regressor_idx);

    regressor_img = arrayfun(@(x) sprintf('beta_%04d.nii',x),regressor_idx,'UniformOutput',false);
end