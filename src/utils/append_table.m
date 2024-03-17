function t = append_table(varargin)
if nargin<2
    error('Need at least two tables as input')
end
    table_lists = varargin(1:nargin);
    t = table_lists{1};
    idx = 2;
    while idx <= numel(table_lists)    
        addon_t = table_lists{idx};
        t = appendtwo(t,addon_t);
        idx = idx + 1;
    end
end
function t = appendtwo(t1,t2)
    vars1 = t1.Properties.VariableNames;
    vars2 = t2.Properties.VariableNames;

    extra1 = vars1(~ismember(vars1,vars2));
    extra2 = vars2(~ismember(vars2,vars1));

    for j = 1:numel(extra1)
        t2.(extra1{j}) = NaN(height(t2),1);
    end
    for j = 1:numel(extra2)
        t1.(extra2{j}) = NaN(height(t1),1);
    end
    t = cat(1,t1,t2);        
end
