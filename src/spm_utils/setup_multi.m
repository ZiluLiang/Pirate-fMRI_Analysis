function [multicond,multicond_fn,multireg_fn] = setup_multi(glm_name,subid,output_dir,gen_multi_cfg,flag_concatenate,preproc_img_dir)
% create the mat file for multiple condtions and multiple regressors
% INPUT
% - glm_name: name of the glm, this will be used to find glm
%             configurations in the glm_configure functions
% - subid: the participant whose mat file is to be generated
% - output_dir: the directory where the mat file is saved
% - gen_multi_cfg: configurations for design matrix, including whether to
%                  use box car or stick function, whether to orthogonalize 
%                  parametric modulators etc. See documentation in
%                  gen_multi
% - flag_concatenate: whether or not to concatenate sessions together
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    [directory,fmri,filepattern]  = get_pirate_defaults(false,'directory','fmri','filepattern');
    glm_config = glm_configure(glm_name);
    glm_dir    = fullfile(directory.fmri_data,glm_config.name);
    
    if nargin<3, output_dir = fullfile(glm_dir,'beh',subid); end
    if nargin<4, gen_multi_cfg = struct();end
    if nargin<5, flag_concatenate = false;end
    if nargin<6, preproc_img_dir = directory.smoothed;end
    checkdir(output_dir)
    
    nii_files  = cellstr(spm_select('FPList',fullfile(preproc_img_dir,subid),[glm_config.filepattern,'.*.nii'])); 
    nuisance_files = cellstr(spm_select('FPList',fullfile(preproc_img_dir,subid),[filepattern.preprocess.nuisance,glm_config.filepattern,'.*.txt']));    
    data_mat   = cellstr(spm_select('FPList',fullfile(directory.fmribehavior,subid),[glm_config.filepattern,'.*mat']));
    if ~flag_concatenate
        % no concatenation, specify mutli-conditions and multi-regressors separately
        multicond          = cell(size(data_mat));
        multicond_fn       = cell(size(data_mat));
        for j = 1:numel(data_mat)
            load(data_mat{j},'data');
            multi   = gen_multiconditions(data,glm_config.conditions,glm_config.pmods,gen_multi_cfg);
            multicond_fn{j} = fullfile(output_dir,sprintf('multi_%d.mat',j));
            multicond{j}    = multi;
            save(multicond_fn{j},'-struct','multi')
            clearvars data multi
        end        
        % multi-regressors: only nuisance regressors are used（headmotion）
        multireg_fn = nuisance_files;
        
        % put reg/cond for different sessions(runs) into different cells
        multireg_fn = cellfun(@(x) cellstr(x),multireg_fn,'UniformOutput',false);
        multicond_fn = cellfun(@(x) cellstr(x),multicond_fn,'UniformOutput',false);
    else        
        % concatenation, concatenate data from all runs together before specifying multi-conditions and multi-regressosrs
        scans = cellfun(@(x) numel(spm_vol(x)),nii_files);
        %multiple conditions
        data_list = cellfun(@(x) load(x,'data').data,data_mat,'UniformOutput',0);
        for j = 1:numel(data_list)
            if ~istable(data_list{j})
                error('data must be in table format')
            end
            variable_names = data_list{j}.Properties.VariableNames;
            onset_variables = variable_names(contains(variable_names,'onset_'));
            for k = 1:numel(onset_variables)
                data_list{j}.(onset_variables{k}) = data_list{j}.(onset_variables{k}) + sum(scans(1:j-1),'all')* fmri.tr;
            end
        end
        data = cat(1,data_list{:});
        multi = gen_multiconditions(data,glm_config.conditions,glm_config.pmods,gen_multi_cfg);
        multicond = {multi};
        multicond_fn = fullfile(output_dir,'multi_concat.mat');
        save(string(multicond_fn),'-struct','multi')
        multicond_fn = {cellstr(multicond_fn)};
        % multi-regressors: currently only nuisance regressors are used（headmotion）
        nuisance_list  = cellfun(@(x) readtable(x),nuisance_files,'UniformOutput',0); %#ok<*UNRCH>
        nuisance_table = cat(1,nuisance_list{:});
        nuisance_file  = fullfile(output_dir,'nuisance_concat.txt');
        writetable(nuisance_table,nuisance_file,'Delimiter',' ','WriteVariableNames',false)
        multireg_fn    = {cellstr(nuisance_file)};
    end
end

function multi = gen_multiconditions(data,cond_names,pmod_names,custom_cfg)
% generate the variables saved to the multi-cond.mat file which can be used
% directly when specifying first level models in SPM.
% example usage: gen_multi(data,names,pmod,cfg) cfg and pmod are optional
% INPUT
%     data: the data table
%     names: names of the event
%     cfg: configurations for design matrix, it has the following fields
%           'use_stick' - whether to use box car or stick function, by default uses stick function (durtation = 0)
%           'tmod',     - the order of time modulation,
%           'orth',     - whether to orthogonalize parametric modulators
%     pmod_names: names of parametric modulators
% Note that this function look for event onset (and/or duration) by looking
% for names of the event/parametric modulator columns in the data table, 
% for event names, onset_eventname/duration_eventname must be a column in the data table
% for pmod names, pmodname must be a column in the data table. For
% instance, a condition called 'stimuli' is set up by looking for columns
% called 'onset_stimuli' and 'duration_stimuli' in the data table. a
% parametric modulator called 'distance' is set up by looking for a column
% called 'distance' in the data table.

% OUTPUT:
%   a struct with the following fields: 'names','onsets','durations','orth','tmod','pmod'.
%   each field value is a cell array or struct array of size 1xN N = numel(eventnames).
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
                     'orth',      0);  % orthogonalise the parametric modulators within the condition or not, by default no.
    
    % if input specifies cfg, combine cfg with default cfg
    if nargin<4, custom_cfg = struct('use_stick',{},'tmod',{},'orth',{}); end
    cfg = repmat(def_cfg,size(cond_names));
    for k = 1:numel(custom_cfg)
        cfg(k) = fill_struct(custom_cfg(k),def_cfg);
    end
    
    % specify conditions
    orth = num2cell([cfg.orth]);
    tmod = num2cell([cfg.tmod]);
    pmod = repmat(struct('name','','param','','poly',''),n_conditions,1);
    onsets    = cell(size(cond_names));
    durations = cell(size(cond_names));

    for j = 1:n_conditions
        % find columns for onset and duration
        onsets{j} = data.(['onset_',cond_names{j}]);
        if ~cfg(j).use_stick
            if ismember(['duration_',cond_names{j}],data.Properties.VariableNames)
                durations{j} = data.(['duration_',cond_names{j}]);
            else
                warning('fail to find column %s in the data table, using stick function instead',['duration_',cond_names{j}])
                durations{j} = zeros(size(onsets{j}));
            end
        else
            durations{j} = zeros(size(onsets{j}));
        end
        
        % filter out nan values
        if numel(onsets{j}) == numel(durations{j})          
            if j<=numel(pmod_names) && ~isempty(pmod_names{j}) % fill in pmod if there are parametric modulators for the current regressor
                % find columns for parametric modulators
                pmod_vals = arrayfun(@(k) data.(pmod_names{j}{k}),1:numel(pmod_names{j}),'UniformOutput',false); 
                % filter out nans
                pmod_nanfilters = cellfun(@(x) ~isnan(x),pmod_vals,'UniformOutput',false); 
                idx = all([~isnan(onsets{j}),~isnan(durations{j}),cat(2,pmod_nanfilters{:})],2);
                % fill in pmod struct
                for k = 1:numel(pmod_names{j})
                    pmod(j).name{k} = pmod_names{j}{k};
                    pmod(j).param{k} = pmod_vals{k}(idx);
                    pmod(j).poly{k} = 1;
                end
            else % if there is no parametric modulator
                idx = all([~isnan(onsets{j}),~isnan(durations{j})],2);
            end
            onsets{j} = onsets{j}(idx);
            durations{j} = durations{j}(idx);
        else
            error('size of onset array and duration array do not match!')
        end

    end
    
    % we filter out empty columns
    nonempty_conds = cellfun(@(x) ~isempty(x),onsets);
    cond_names  = cond_names(nonempty_conds);
    onsets      = onsets(nonempty_conds);
    durations   = durations(nonempty_conds);
    orth        = orth(nonempty_conds);
    tmod        = tmod(nonempty_conds);
    pmod        = pmod(nonempty_conds);
    multi = cell2struct({cond_names,onsets,durations,orth,tmod,pmod}',...
                        {'names','onsets','durations','orth','tmod','pmod'});
end