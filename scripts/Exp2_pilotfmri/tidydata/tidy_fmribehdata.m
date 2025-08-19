% this files organizes behavioral data during fmri scans into the the
% format for univariate first level analysis. renamed files are in the same convention as
% renaed fmri files, which is:
% sub-x_task-taskname_run-j.mat
% taskname is 'piratenavigation' or 'localizer', x is subid, j is run
% each mat file contains a table with onset_eventname and duration_eventname
% 
% if an event did not happen in a given trial, the value of onset and duration
% in the corresponding row will be set to nan.


untidied_behavior_pardir = 'E:\pirate_fmri\Analysis\data\Exp2_pilotfmri\untidiedbhavior';
untidied_behaviordir = fullfile(untidied_behavior_pardir,'fmri_behavior');
renamer_fn = fullfile('E:\pirate_fmri\Analysis\data\Exp2_pilotfmri','renamer.json');
renamer    = loadjson(renamer_fn);
participants = get_pirate_defaults(false,'participants');
output_dir = 'E:\pirate_fmri\Analysis\data\Exp2_pilotfmri\fmri\beh';
ids = participants.validids;
checkdir(fullfile(output_dir,ids))

%% move stimuli file to the folder
for isub = 1:numel(ids)
    fprintf('copying stimuli file %s\n',ids{isub})
    exptid = renamer.(ids{isub}){1};
    stim_data_pat = sprintf('stimlist_%s.txt',exptid);
    new_stimdata_fn = sprintf('sub-%s_stimlist.txt',...
                            strrep(ids{isub},'sub',''));
    old_fn = fullfile(untidied_behavior_pardir,'param_ptb',stim_data_pat);
    new_fn = fullfile(output_dir,ids{isub},new_stimdata_fn);
    copyfile(old_fn,new_fn)
    clearvars exptid stim_data_pat new_stimdata_fn old_fn new_fn
end

%% navigation task
taskname = "maintask";
for isub = 1:numel(ids)
    fprintf('tidying navigation task %s\n',ids{isub})
    for run = 1:6
        exptid = renamer.(ids{isub}){1};
        bhav_data_pat = sprintf('^%s_%s_run%d_.*.mat',taskname,exptid,run);
        meta_data_pat = sprintf('^meta_%s_%s_run%d_.*.mat',taskname,exptid,run);
        load(spm_select('FPList',untidied_behaviordir,bhav_data_pat),'data');
        load(spm_select('FPList',untidied_behaviordir,meta_data_pat),'meta');
        olddata = data;        
        
        % note that onset_trial in the original data table was from pre-generated task sequence
        % time_* is the time aligned to the scanner start
        % so there is usually and offset between first onset_trial and
        % time_cue because in the fmri task code we added some pre-task
        % lead in time to allow for scanner signal to settle in and have
        % participants to be mentally prepared for the task

        % first we get the timimg info for building GLMs:
        % for resp trials: stimuli -> delay -> probe(until response or timeout -> inter-trial jitter
        % for noresp trials: stimuli -> inter-trial jitter
        olddata.onset_stimuli    = olddata.time_cue; 
        olddata.onset_probe      = olddata.time_probe;                
        olddata.onset_response   = olddata.time_response;
        for t = 1:size(olddata,1)
            if isnan(olddata.ctrl_resp(t))
                olddata.duration_stimuli(t) = olddata.time_delay(t) - olddata.time_cue(t);
                if isnan(olddata.time_response(t)) % participant did not make a response in time
                    olddata.duration_probe(t)   = olddata.time_jitter(t) - olddata.time_probe(t);
                else % participant made a response in time
                    olddata.duration_probe(t)   = olddata.time_response(t) - olddata.time_probe(t);
                end
                % this is to help check distribution of timing
                olddata.duration_delay(t)   = olddata.time_probe(t) - olddata.time_delay(t);
            else
                olddata.duration_stimuli(t) = olddata.time_jitter(t) - olddata.time_cue(t);
                olddata.duration_probe(t)   = nan;
                olddata.duration_delay(t)   = nan;
            end
        end
        olddata.duration_iti  = [olddata.time_cue(2:end) - olddata.time_jitter(1:end-1); nan];
        olddata.resp_rt = olddata.time_response - olddata.time_probe;

        %trial difficulty
        olddata.TD_eucdist = sqrt(vecnorm(([olddata.optionT_x,olddata.optionT_y]-[olddata.optionD_x,olddata.optionD_y])')');
        olddata.TD_cbdist  = sum(abs([olddata.optionT_x,olddata.optionT_y]-[olddata.optionD_x,olddata.optionD_y]),2);
        

        %hierachy model: stimuli location wrt screen centre
        olddata.x_sign = sign(olddata.stim_x);
        olddata.x_dist = abs(olddata.stim_x);
        olddata.y_sign = sign(olddata.stim_y);
        olddata.y_dist = abs(olddata.stim_y);


        for t = 1:size(olddata,1)
            % construct regressors for repetition suppression analysis
            if t == 1 || olddata.ctrl_resp(t-1) % exclude from repetition suppression analysis if in trial 1 or if response is required in the previous trial
                % distance based on 10d model (feature-based)
                olddata.featuredist(t)   = nan;
                olddata.leftdist(t)     = nan;
                olddata.rightdist(t)     = nan;
                % distance based on 2d model (groundtruth or response)
                olddata.dist2d(t)        = nan;
                olddata.dist2d_resp(t)   = nan;
                % distance based on hierachy model
                olddata.hrchydist_ucord(t) = nan; 
                olddata.hrchydist_quadr(t) = nan; 

                olddata.rstrials(t)      = nan;
                olddata.excluders(t)     = 1; 
            else
                % distance based on 10d model (feature-based)
                olddata.leftdist(t)    = 1-strcmp(olddata.stim_left(t),olddata.stim_left(t-1));
                olddata.rightdist(t)   = 1-strcmp(olddata.stim_right(t),olddata.stim_right(t-1));
                olddata.featuredist(t) = olddata.leftdist(t)+olddata.rightdist(t);

                % distance based on 2d model (groundtruth or response)
                olddata.dist2d(t) =  vecnorm([olddata.stim_x(t),olddata.stim_y(t)]-[olddata.stim_x(t-1),olddata.stim_y(t-1)]);               
                
                % distance based on hierachy model                
                olddata.hrchydist_ucord(t) = vecnorm([olddata.x_dist(t),olddata.y_dist(t)]-[olddata.x_dist(t-1),olddata.y_dist(t-1)]); % distance based on hierachy model: unsigned distance hierachy
                olddata.hrchydist_quadr(t) = vecnorm([olddata.x_sign(t),olddata.y_sign(t)]-[olddata.x_sign(t-1),olddata.y_sign(t-1)]); % distance based on hierachy model: quadrant/axis

                olddata.rstrials(t)    = 1;
                olddata.excluders(t)   = nan;                
            end
        end
        
            
        olddata.onset_excluders    = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials     = olddata.rstrials .* olddata.onset_stimuli;
        olddata.duration_excluders = olddata.excluders .* olddata.duration_stimuli;
        olddata.duration_rstrials  = olddata.rstrials .* olddata.duration_stimuli;
        
        % construct regressors for stimulitype analysis
        stype_filters = struct( ...
            'training',        strcmp(olddata.stim_group,'training'), ...
            'validation',      strcmp(olddata.stim_group,'validation'), ...
            'testcenter',      strcmp(olddata.stim_group,'testcenter'), ...
            'TMtestnoncenter', strcmp(olddata.stim_group,'testnoncenter')&olddata.stim_map<2, ...
            'CMtestnoncenter', strcmp(olddata.stim_group,'testnoncenter')&olddata.stim_map>=2 ...
            );
        stypes = fieldnames(stype_filters);
        st_onsets = repmat({nan(100,1)},1,numel(stypes));
        st_durats = repmat({nan(100,1)},1,numel(stypes));
        for jst = 1:numel(stypes)
            st_onsets{jst}(stype_filters.(stypes{jst})) = olddata.onset_stimuli(stype_filters.(stypes{jst}));
            st_durats{jst}(stype_filters.(stypes{jst})) = olddata.duration_stimuli(stype_filters.(stypes{jst}));
        end
        stonset_vars    = cellfun(@(x) sprintf('onset_%s',x),stypes,'uni',0);
        stduration_vars = cellfun(@(x) sprintf('duration_%s',x),stypes,'uni',0);
        olddata       = addvars(olddata,st_onsets{:},'NewVariableNames',stonset_vars);
        olddata       = addvars(olddata,st_durats{:},'NewVariableNames',stduration_vars);
    
        ctrl_resp = olddata.ctrl_resp;
        ctrl_resp(ctrl_resp==0) = nan;
        ctrl_noresp = 1-olddata.ctrl_resp;
        ctrl_noresp(ctrl_noresp==0) = nan;
        olddata.onset_resp    = olddata.onset_stimuli.*ctrl_resp;
        olddata.duration_resp = olddata.duration_stimuli.*ctrl_resp;
        olddata.onset_noresp    = olddata.onset_stimuli.*ctrl_noresp;
        olddata.duration_noresp = olddata.duration_stimuli.*ctrl_noresp;
        
        % construct regressors for each stimulus as an individual event
        unique_ids = unique(olddata.stim_id);
        for j = 1:numel(unique_ids)
            sid = unique_ids(j);
            olddata.(sprintf('onset_stim%02d',sid)) = (olddata.stim_id==sid) .* olddata.onset_stimuli;
            olddata.(sprintf('onset_stim%02d',sid))(olddata.stim_id~=sid) = nan;
            olddata.(sprintf('duration_stim%02d',sid)) = (olddata.stim_id==sid) .* olddata.duration_stimuli;
            olddata.(sprintf('duration_stim%02d',sid))(olddata.stim_id~=sid) = nan;
        end

        % construct regressors for stimuli at each location as an individual event
        unique_lids = unique(olddata.stim_locid);
        for j = 1:numel(unique_lids)
            lid = unique_lids(j);
            olddata.(sprintf('onset_loc%02d',lid)) = (olddata.stim_locid==lid) .* olddata.onset_stimuli;
            olddata.(sprintf('onset_loc%02d',lid))(olddata.stim_locid~=lid) = nan;
            olddata.(sprintf('duration_loc%02d',lid)) = (olddata.stim_locid==lid) .* olddata.duration_stimuli;
            olddata.(sprintf('duration_loc%02d',lid))(olddata.stim_locid~=lid) = nan;
        end
        
        keep_v = {'stim_id','stim_locid','stim_group','stim_left','stim_right','stim_leftgroup','stim_leftgroup','stim_img',...% fields in the orginal data table
                  'stim_x','stim_y','stim_attrx','stim_attry','resp_x','resp_y','resp_choice','resp_acc','resp_rt',...% fields in the orginal data table
                  'stim_map','delay','iti','duration_delay','duration_iti',...
                  'x_dist','x_sign','y_dist','y_sign','hrchydist_ucord','hrchydist_quadr',...% hierachy model: stimuli location wrt screen centre
                  'onset_stimuli', 'duration_stimuli',... % stimuli event
                  'onset_probe',   'duration_probe',... % stimuli event
                  'onset_response','duration_response',... % response event                
                  'onset_rstrials','duration_rstrials',...  % repetition suppression event
                  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
                  'dist2d','featuredist','leftdist','rightdist',... % parametric modulators for repetition suppression:groundtruth distance/recontructed distance from participant response between current stimulus and previous stimulus 
                  'TD_eucdist','TD_cbdist'...
                  };
        keep_v = [keep_v,...
                  stonset_vars',stduration_vars',...
                  arrayfun(@(x) sprintf('onset_stim%02d',x),unique_ids,'uni',0)',...% separated stimulus event onset
                  arrayfun(@(x) sprintf('duration_stim%02d',x),unique_ids,'uni',0)',...%#ok<*AGROW> % separated stimulus event duration
                  arrayfun(@(x) sprintf('onset_loc%02dresp',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('duration_loc%02dresp',x),unique_ids,'uni',0)'...
                 ]; 
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-pirate2AFC_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        save(fullfile(output_dir,ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-pirate2AFC_run-%d.mat',...
                                 strrep(ids{isub},'sub',''), run);
        checkdir(fullfile(output_dir,'metadata'))
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
        clearvars exptid bhav_data_pat meta_data_pat olddata data keep_v remove_v
    end
end

%% localizer task
taskname = "localizer";
for isub = 1:numel(ids)
    fprintf('tidying localizer task %s\n',ids{isub})
    for run = 1
        exptid = renamer.(ids{isub}){1};
        bhav_data_pat = sprintf('^%s_%s_run%d_.*.mat',taskname,exptid,run);
        meta_data_pat = sprintf('^meta_%s_%s_run%d_.*.mat',taskname,exptid,run);
        load(spm_select('FPList',untidied_behaviordir,bhav_data_pat),'data');
        load(spm_select('FPList',untidied_behaviordir,meta_data_pat),'meta');
        olddata = data; 
        
        olddata.onset_stimuli    = olddata.time_cue; 
        olddata.onset_probe      = olddata.time_probe;                
        olddata.onset_response   = olddata.time_response;
        for t = 1:size(olddata,1)
            if isnan(olddata.ctrl_resp(t))
                olddata.duration_stimuli(t) = olddata.time_delay(t) - olddata.time_cue(t);
                if isnan(olddata.time_response(t)) % participant did not make a response in time
                    olddata.duration_probe(t)   = olddata.time_jitter(t) - olddata.time_probe(t);
                else % participant made a response in time
                    olddata.duration_probe(t)   = olddata.time_response(t) - olddata.time_probe(t);
                end
                % this is to help check distribution of timing
                olddata.duration_delay(t)   = olddata.time_probe(t) - olddata.time_delay(t);
            else
                olddata.duration_stimuli(t) = olddata.time_jitter(t) - olddata.time_cue(t);
                olddata.duration_probe(t)   = nan;
                olddata.duration_delay(t)   = nan;
            end
        end
        olddata.duration_iti  = [olddata.time_cue(2:end) - olddata.time_jitter(1:end-1); nan];
        olddata.resp_rt = olddata.time_response - olddata.time_probe;

        %trial difficulty
        olddata.TD_eucdist = sqrt(vecnorm(([olddata.optionT_x,olddata.optionT_y]-[olddata.optionD_x,olddata.optionD_y])')');
        olddata.TD_cbdist  = sum(abs([olddata.optionT_x,olddata.optionT_y]-[olddata.optionD_x,olddata.optionD_y]),2);

        
        
        % construct regressors for repetition suppression analysis
        for t = 1:size(olddata,1)
            if t == 1 || olddata.ctrl_resp(t-1) % exclude from repetition suppression analysis if in trial 1 or if it previous one is a response trial
                olddata.dist2d(t) =  nan;
                olddata.excluders(t)  = 1;
                olddata.rstrials(t)   = nan;
                olddata.prev_stimx(t) = nan;
                olddata.prev_stimy(t) = nan;
            else
                olddata.excluders(t)  = nan;
                olddata.rstrials(t)   = 1;
                olddata.dist2d(t)     =  sqrt(vecnorm([olddata.stim_x(t),olddata.stim_y(t)]-[olddata.stim_x(t-1),olddata.stim_y(t-1)]));
                olddata.prev_stimx(t) = olddata.stim_x(t-1);
                olddata.prev_stimy(t) = olddata.stim_y(t-1);
            end
        end
        olddata.onset_excluders = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials  = olddata.rstrials .* olddata.onset_stimuli;
        olddata.duration_excluders = olddata.excluders .* olddata.duration_stimuli;
        olddata.duration_rstrials  = olddata.rstrials .* olddata.duration_stimuli;
        
        % construct regressors for stimuli at each location as an individual event
        unique_lids = unique(olddata.stim_locid);
        for j = 1:numel(unique_lids)
            lid = unique_lids(j);
            olddata.(sprintf('onset_loc%02d',lid)) = (olddata.stim_locid==lid) .* olddata.onset_stimuli;
            olddata.(sprintf('onset_loc%02d',lid))(olddata.stim_locid~=lid) = nan;
            olddata.(sprintf('duration_loc%02d',lid)) = (olddata.stim_locid==lid) .* olddata.duration_stimuli;
            olddata.(sprintf('duration_loc%02d',lid))(olddata.stim_locid~=lid) = nan;
        end

        olddata.stim_attrx = olddata.stim_attrx+1;
        olddata.stim_attry = olddata.stim_attry+1;

        keep_v = {'stim_locid','stim_group',...% fields in the orginal data table
                  'stim_x','stim_y','stim_attrx','stim_attry','resp_x','resp_y','resp_choice','resp_acc','resp_rt',...% fields in the orginal data table
                  'delay','iti','duration_delay','duration_iti',...
                  'onset_stimuli', 'duration_stimuli',... % stimuli event
                  'onset_probe',   'duration_probe',... % stimuli event
                  'onset_response','duration_response',... % response event                
                  'onset_rstrials','duration_rstrials',...  % repetition suppression event
                  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
                  'dist2d','featuredist','leftdist','rightdist',... % parametric modulators for repetition suppression:groundtruth distance/recontructed distance from participant response between current stimulus and previous stimulus 
                  'TD_eucdist','TD_cbdist'...
                  };
        keep_v = [keep_v,...
                  arrayfun(@(x) sprintf('onset_loc%02d',x),unique_ids,'uni',0)',...#
                  arrayfun(@(x) sprintf('duration_loc%02d',x),unique_ids,'uni',0)'];              
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-localizer2AFC_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        save(fullfile(output_dir,ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-localizer2AFC_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
        clearvars exptid bhav_data_pat meta_data_pat olddata data keep_v remove_v
    end
end


%% symballoddball task
taskname = "symoddball";
for isub = 1:numel(ids)
    fprintf('tidying symoddball task %s\n',ids{isub})
    for run = 1:2
        exptid = renamer.(ids{isub}){1};
        bhav_data_pat = sprintf('^%s_%s_run%d_.*.mat',taskname,exptid,run);
        meta_data_pat = sprintf('^meta_%s_%s_run%d_.*.mat',taskname,exptid,run);
        load(spm_select('FPList',untidied_behaviordir,bhav_data_pat),'data');
        load(spm_select('FPList',untidied_behaviordir,meta_data_pat),'meta');
        olddata = data; 
        
        olddata.onset_stimuli    = olddata.time_cue; 
        olddata.onset_response   = olddata.time_response;
        olddata.duration_stimuli = olddata.time_jitter - olddata.time_cue;
        olddata.duration_iti     = [olddata.time_cue(2:end) - olddata.time_jitter(1:end-1);nan];
        olddata.resp_rt = olddata.time_response - olddata.time_cue;

        olddata.sym_axloc = mod(olddata.sym_id-1,5)-2;
        olddata.sym_axloc(olddata.isnovel) = nan;
        olddata.sym_map = mod(ceil(olddata.sym_id/5),2);
        olddata.sym_map(olddata.isnovel) = nan;
        
        % construct regressors for repetition suppression analysis
        for t = 1:size(olddata,1)
            if t == 1 || olddata.isnovel(t-1) % exclude from repetition suppression analysis if in trial 1 or if the previous one is a novel symbol
                olddata.dist2d(t)       =  nan;
                olddata.distmap(t)      =  nan;
                olddata.distPTAaxis(t)  =  nan;
                olddata.distPTAaxloc(t) =  nan;
                olddata.excluders(t)  = 1;
                olddata.rstrials(t)   = nan;
            else
                olddata.distmap(t)      =  olddata.sym_map(t)~=olddata.sym_map(t-1);
                olddata.distPTAaxis(t)  =  1-strcmp(olddata.sym_side{t},olddata.sym_side{t-1});
                olddata.dist1dPTAaxloc(t) =  abs(olddata.sym_axloc(t)-olddata.sym_axloc(t-1));
                olddata.dist5dPTAaxloc(t) =  olddata.sym_axloc(t)~=olddata.sym_axloc(t-1);
                if olddata.distPTAaxis(t)
                    olddata.dist2d(t)       =  sqrt(olddata.sym_axloc(t)+olddata.sym_axloc(t-1)); % need to rethink about this one
                else
                    olddata.dist2d(t)       =  olddata.dist1dPTAaxloc(t);
                end
                olddata.excluders(t)  = nan;
                olddata.rstrials(t)   = 1;
            end
        end
        olddata.onset_excluders = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials  = olddata.rstrials .* olddata.onset_stimuli;
        olddata.duration_excluders = olddata.excluders .* olddata.duration_stimuli;
        olddata.duration_rstrials  = olddata.rstrials .* olddata.duration_stimuli;
        
        isnovel = 1*olddata.isnovel;
        islearned = 1-isnovel;
        isnovel(olddata.isnovel==0) = nan;
        islearned(olddata.isnovel==1) = nan;
        
        olddata.onset_learnedsym    = olddata.onset_stimuli .* islearned;
        olddata.duration_learnedsym = olddata.onset_stimuli .* islearned;
        olddata.onset_novelsym      = olddata.onset_stimuli .* isnovel;
        olddata.duration_novelsym   = olddata.onset_stimuli .* isnovel;

        % construct regressors for stimuli at each location as an individual event
        unique_sids = unique(olddata.sym_id);
        for j = 1:numel(unique_sids)
            sid = unique_sids(j);
            olddata.(sprintf('onset_sym%02d',sid)) = (olddata.sym_id==sid) .* olddata.onset_stimuli;
            olddata.(sprintf('onset_sym%02d',sid))(olddata.sym_id~=sid) = nan;
            olddata.(sprintf('duration_sym%02d',sid)) = (olddata.sym_id==sid) .* olddata.duration_stimuli;
            olddata.(sprintf('duration_sym%02d',sid))(olddata.sym_id~=sid) = nan;
        end

        keep_v = {'sym_id','sym_side','sym_axloc','sym_map',...% fields in the orginal data table
                  'sym_img','ctrl_resp','resp_choice','isnovel','resp_acc','resp_y','resp_rt',...% fields in the orginal data table
                  'iti','duration_iti',...
                  'onset_stimuli', 'duration_stimuli',... % stimuli event
                  'onset_response',... % response event                
                  'onset_learnedsym','onset_novelsym','duration_learnedsym','duration_novelsym',...
                  'onset_rstrials','duration_rstrials',...  % repetition suppression event
                  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
                  'dist2d','distmap','distPTAaxis','dist1dPTAaxloc','dist5dPTAaxloc'... % parametric modulators for repetition suppression
                  };
        keep_v = [keep_v,...
                  arrayfun(@(x) sprintf('onset_sym%02d',x),unique_ids,'uni',0)',...#
                  arrayfun(@(x) sprintf('duration_sym%02d',x),unique_ids,'uni',0)'];              
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-symboloddball_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        save(fullfile(output_dir,ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-symboloddball_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
        clearvars exptid bhav_data_pat meta_data_pat olddata data keep_v remove_v
    end
end
