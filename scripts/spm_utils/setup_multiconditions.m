function [M,M_fn] = setup_multiconditions(glm_name,subid,output_dir,gen_multi_cfg)
% create the mat file for multiple condtions
% INPUT
% - glm_name: name of the glm, this will be used to find glm
%             configurations in the get_glm_config functions
% - subid: the participant whose mat file is to be generated
% - output_dir: the directory where the mat file is saved
% - gen_multi_cfg: configurations for design matrix, including whether to
%                  use box car or stick function, whether to orthogonalize 
%                  parametric modulators etc. See documentation in
%                  gen_multi
    directory  = get_pirate_defaults(false,'directory');
    glm_config = get_glm_config(glm_name);
    glm_dir    = fullfile(directory.fmri_data,glm_config.name);
    
    if nargin<3, output_dir = fullfile(glm_dir,'beh',subid); end
    checkdir(output_dir)
    
    data_mat   = cellstr(spm_select('FPList',fullfile(directory.fmribehavior,subid),[glm_config.filepattern,'.*mat']));
    M          = cell(size(data_mat));
    M_fn       = cell(size(data_mat));
    for j = 1:numel(data_mat)
        load(data_mat{j},'data');
        multi   = gen_multi(data,glm_config.conditions,glm_config.pmods,gen_multi_cfg);
        M_fn{j} = fullfile(output_dir,sprintf('multi_%d.mat',j));
        M{j}    = multi;
        save(M_fn{j},'-struct','multi')
        clearvars data multi
    end
end

function multi = gen_multi(data,cond_names,pmod_names,custom_cfg)
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
% pmod      = struct('name',{},...
%                    'param',{},...
%                    'poly',{});
% pmod(1).name{1} = 'pm_reg1_1';
% pmod(1).param{1} = [1,2,3,4,5];
% pmod(1).poly{1} = 1;
% 
% pmod(2).name{1} = 'pm_reg2_1';
% pmod(2).param{1} = [1,2,3,4,5];
% pmod(2).poly{1} = 1;
    
    n_conditions = numel(cond_names);
    cond_names = reshape(cond_names,[1,n_conditions]); % flatten input just in case 
    
    % default configurations
    def_cfg = struct('use_stick', true,... %by default uses stick function (durtation = 0)
                     'tmod',      0,... % the order of time modulation,
                     'orth',      1); % orthogonalise the parametric modulators within the condition or not
    
    % if input specifies cfg, combine cfg with default cfg
    if nargin<4, custom_cfg = struct('use_stick',{},'tmod',{},'orth',{}); end
    cfg = repmat(def_cfg,n_conditions,1);
    for k = 1:numel(custom_cfg)
        cfg(k) = fill_struct(custom_cfg(k),def_cfg);
    end
    
    % specify conditions
    orth = num2cell([cfg.orth]);
    tmod = num2cell([cfg.tmod]);
    pmod = repmat(struct('name',{},'param',{},'poly',{}),n_conditions,1);
    try
        if ~isempty(pmod_names)
            for j = 1:numel(cond_names)
                if j>numel(pmod_names) && isempty(pmod_names{j})
                    pmod(j) = struct('name',{},'param',{},'poly',{});
                else
                    for k = 1:numel(pmod_names{j})
                        pmod(j).name{k} = pmod_names{j}{k};
                        data_col = data.(pmod_names{j}{k});
                        pmod_val = data_col(~isnan(data_col));
                        pmod(j).param{k} = pmod_val;
                        pmod(j).poly{k} = 1;
                    end
                end
            end
        end
    catch
        pmod = struct('name',{},'param',{},'poly',{});
    end


    onsets    = cell(size(cond_names));
    durations = cell(size(cond_names));

    for j = 1:numel(cond_names)
        % find columns for onset and duration
        onsets{j} = data.(['onset_',cond_names{j}]);
        if ~cfg(j).use_stick
            if ismember(['duration_',cond_names{j}],data.Properties.VariableNames)
                durations{j} = data.(['duration_',cond_names{j}]);
            else
                warning('fail to find column %s in the data table, using stick function instead',['duration_',cond_names{j}])
                durations{j} = 0;
            end
        else
            durations{j} = 0;
        end
        % filter out nan values
        if numel(onsets{j}) == numel(durations{j})             
            idx = all([~isnan(onsets{j}),~isnan(durations{j})],2);
            onsets{j} = onsets{j}(idx);
            durations{j} = durations{j}(idx);
        elseif durations{j}==0
            onsets{j} = onsets{j}(~isnan(onsets{j}));
        else
            error('size of onset array and duration array do not match!')
        end
        % find columns for parametric modulators
        if j<numel(pmod_names) && ~isempty(pmod_names{j})
            for k = 1:numel(pmod_names{j})
                pmod(j).name{k} = pmod_names{j}{k};
                data_col = data.(pmod_names{j}{k});
                pmod_val = data_col(~isnan(data_col));
                pmod(j).param{k} = pmod_val;
                pmod(j).poly{k} = 1;
            end
        end
    end
    
    multi = cell2struct({cond_names,onsets,durations,orth,tmod,pmod}',...
                        {'names','onsets','durations','orth','tmod','pmod'});
end