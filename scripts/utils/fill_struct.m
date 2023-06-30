function s = fill_struct(s,s0)
% fill empty or non-existing fields in s with values from s0
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    f_s0 = fieldnames(s0);
    f_s  = fieldnames(s);
    for j = 1:numel(f_s0)
        if ~ismember(f_s0{j},f_s) || isempty(s.(f_s0{j}))
            s.(f_s0{j})  = s0.(f_s0{j});
        end
    end
end