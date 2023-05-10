function T = subtract_table(A,B)
% subtract all numeric fields of the table and return the result as a table
% INPUT: A,B must be tables of with same variables
error_flag = true;
if istable(A)&&istable(B)
    var_A = A.Properties.VariableNames;
    var_B = B.Properties.VariableNames;
    if all([cellfun(@(x) ismember(x,var_B),var_A),cellfun(@(x) ismember(x,var_A),var_B)])
        error_flag = false;
        vars = var_A;
        T = table;
    end
end
if error_flag
    error('Invalid inputs')
end

for j = 1:numel(vars)
    if isnumeric(A.(vars{j})) && isnumeric(B.(vars{j}))
        T.(vars{j}) = A.(vars{j}) - B.(vars{j});
    end
end

end