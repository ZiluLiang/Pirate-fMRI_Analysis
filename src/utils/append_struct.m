function s = append_struct(varargin)
%Usage: append_struct(s,addon_s1,addon_s2,...,addon_sn,append_opt)

%check inputs
if nargin<3
    error(['Number of arguments is not enough. ',...
           'At least two structs should be enter as input argument']);
else
    append_opt = varargin{nargin};
    struct_lists = varargin(1:nargin-1);
end
valid_append_opt = {'concatenate','add_field','expand'};
if ~contains(append_opt,valid_append_opt)
    error(['Invalid append option, should be one of :',...
        join(string(valid_append_opt),',')])
end
check_struct = cellfun(@(x) isstruct(x),struct_lists);
if ~all(check_struct)
    error('All but last inputs must be structs!');
end

%append structs
s = struct_lists{1};
idx = 2;
while idx <= numel(struct_lists)    
    addon_s = struct_lists{idx};
    s = append_s(s,addon_s,append_opt);
    idx = idx + 1;
end

function appended_s = append_s(s,addon_s,append_opt)
    [m0,n0] = size(s);
    [m,n]   = size(addon_s);

    f0 = fieldnames(s);
    f  = fieldnames(addon_s);

    appended_s = s;

    switch append_opt
        case 'concatenate'
            % if contains same fields ,then concatenate the two structs together
            if isequal(numel(f0),numel(f)) && all(contains(f0,f))
                if m0==m && n0==n
                    appended_s = [s;addon_s];
                elseif m0~=m && n0~=n
                    warning(['Dimension of input structures mismatch,',...
                            'proceed with reshaping them into column vector'])
                    appended_s = [reshape(s,[],1);reshape(addon_s,[],1)];
                else
                    cat_dim = find([m0==m, n0==n]== 0);
                    appended_s = cat(cat_dim,s,addon_s);
                end
            end
        case 'add_field'
            % Add newfields to each element in s            
            checkdim(s,addon_s)
            for j = 1:numel(f)
                if ~isfield(s,f{j})
                    for k = 1:numel(s)
                        appended_s(k).(f{j}) = addon_s(k).(f{j});
                    end
                end
            end
        case 'expand'
            % Add newfields to s but in new elements
            for j = 1:numel(addon_s)
                for k = 1:numel(f)
                    appended_s(numel(s)+j).(f{k}) = addon_s(j).(f{k});
                end
            end
    end
end
end