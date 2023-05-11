function [M,M_fn] = setup_multiconditions(glm_name,subid)
% create the mat file for multiple condtions
    directory  = get_pirate_defaults(false,'directory');
    glm_config = get_glm_config(glm_name);
    glm_dir    = fullfile(directory.fmri_data,glm_config.name);
    output_dir = fullfile(glm_dir,'first',subid);
    checkdir(output_dir)
    
    search_str = sprintf(glm_config.filepattern,strrep(subid,'sub',''));
    data_mat   = cellstr(spm_select('FPList',directory.fmribehavior,search_str));
    M          = cell(size(data_mat));
    M_fn       = cell(size(data_mat));
    for j = 1:numel(data_mat)
        load(data_mat{j},'data');
        multi   = gen_multi(data,glm_config.conditions);
        M_fn{j} = fullfile(output_dir,sprintf('multi_%d.mat',j));
        M{j}    = multi;
        save(M_fn{j},'-struct','multi')
        clearvars data multi
    end
end

function multi = gen_multi(data,names,varargin)
% generate the variables saved to the multi-cond.mat file which can be used
% directly when specifying first level models in SPM.
% example usage: gen_multi(data,names,cfg,pmod) cfg and pmod are optional
% INPUT
%     data: the data table
%     names: names of the event
%     cfg: 
%     pmod:
% OUTPUT: 1xN N = numel(eventnames)
%    cell arrays - names,onsets,durations,orth,tmod
%    struct array - pmod
% Output format are constructed according to the spm documentation, e.g.
% names     = {'stimuli','response'};% names of the conditions
% onsets    = {[1,4,7],...           % onsets of the conditions
%              [2,5,8]};
% durations = {[1,1,1],...           % durations of each instance
%              [0,0,0]};
% orth      = {1,1};    
% tmod      = {0,0};
% pmod      = struct(...
%                     'name',{},...
%                     'param',{},...
%                     'poly',{}...
%                    );
% pmod(1).name{1} = 'pm_reg1_1';
% pmod(1).param{1} = [1,2,3,4,5];
% pmod(1).poly{1} = 1;
% 
% pmod(2).name{1} = 'pm_reg2_1';
% pmod(2).param{1} = [1,2,3,4,5];
% pmod(2).poly{1} = 1;
    
    n_conditions = numel(names);
    names = reshape(names,[1,n_conditions]); % flatten input just in case 
    
    % default configurations
    def_cfg = struct('use_stick', true,... %by default uses stick function (durtation = 0)
                     'tmod',     0,... % the order of time modulation,
                     'orth',     1); % orthogonalise the parametric modulators within the condition or not
    
    % if input specifies cfg, combine cfg with default cfg
    cfg = repmat(def_cfg,n_conditions,1);
    if numel(varargin) >= 1
        for k = 1:numel(varargin{1})
            cfg(k) = fill_struct(varargin{1}(k),def_cfg);
        end
        if numel(varargin)>1
            warning('ignoring excessive inputs')
        end
    end
    
    % specify conditions
    orth = num2cell([cfg.orth]);
    tmod = num2cell([cfg.tmod]);
    pmod = struct('name',{},'param',{},'poly',{});% TODO: specification of pmod
    onsets    = cell(size(names));
    durations = cell(size(names));

    for j = 1:numel(names)
        onsets{j} = data.(['onset_',names{j}]);
        onsets{j} = onsets{j}(~isnan(onsets{j})); % filter out nan values
        if ~cfg(j).use_stick && ismember(['duration_',names{j}],data.Properties.VariableNames)
            durations{j} = data.(['duration_',names{j}]);
            durations{j} = durations{j}(~isnan(durations{j})); % filter out nan values
        else
            durations{j} = 0;
        end
    end
    
    multi = cell2struct({names,onsets,durations,orth,tmod,pmod}',...
                        {'names','onsets','durations','orth','tmod','pmod'});
end