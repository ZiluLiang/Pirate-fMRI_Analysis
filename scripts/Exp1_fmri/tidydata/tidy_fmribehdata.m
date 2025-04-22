% this files organizes behavioral data during fmri scans into the the
% format for univariate first level analysis. renamed files are in the same convention as
% renaed fmri files, which is:
% sub-x_task-taskname_run-j.mat
% taskname is 'piratenavigation' or 'localizer', x is subid, j is run
% each mat file contains a table with onset_eventname and duration_eventname
% 
% if an event did not happen in a given trial, the value of onset and duration
% in the corresponding row will be set to nan.

clear;clc;
untidied_behavior_pardir = 'E:\pirate_fmri\Analysis\data\Exp1_fmri\untidiedbhavior';
untidied_behaviordir = fullfile(untidied_behavior_pardir,'fmri_behavior');
renamer_fn = fullfile('E:\pirate_fmri\Analysis\data\Exp1_fmri','renamer.json');
renamer    = loadjson(renamer_fn);
participants = get_pirate_defaults(false,'participants');
output_dir = 'E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\beh';
ids = participants.validids;%participants.cohort2ids; %participants.validids
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
    for run = 1:4
        exptid = renamer.(ids{isub}){1};
        bhav_data_pat = sprintf('^%s_%s_run%d_.*.mat',taskname,exptid,run);
        meta_data_pat = sprintf('^meta_%s_%s_run%d_.*.mat',taskname,exptid,run);
        load(spm_select('FPList',untidied_behaviordir,bhav_data_pat),'data');
        load(spm_select('FPList',untidied_behaviordir,meta_data_pat),'meta');
        olddata = data;        

        color_shape = cellfun(@(x) strsplit(strrep(x,'.png',''),'_'),olddata.stim_img,'UniformOutput',false);
        color_shape = cat(1,color_shape{:});
        olddata.stim_color = color_shape(:,1);
        olddata.stim_shape = color_shape(:,2);
        olddata.onset_stimuli    = olddata.time_cue - meta.time.task_start;
        for t = 1:size(olddata,1)
            if isnan(olddata.time_arena(t))
                olddata.duration_stimuli(t) = olddata.time_jitter(t) - olddata.time_cue(t);
            else
                olddata.duration_stimuli(t) = olddata.time_arena(t) - olddata.time_cue(t);
            end
        end
        olddata.onset_response    = olddata.time_arena - meta.time.task_start;
        olddata.duration_response = olddata.time_jitter - olddata.time_arena;
        
        % recontruct participant's map of stimuli based on their response
        % and calculate pairwise distance between stimuli for the current
        % run
        resp_map = [olddata.stim_id, olddata.resp_x, olddata.resp_y];
        resp_map = array2table(resp_map(olddata.ctrl_resp==1,:),'VariableNames',{'stim_id','resp_x','resp_y'});
        resp_map_fn = sprintf('sub-%s_task-piratenavigation_run-%d.csv',...
                                strrep(ids{isub},'sub',''), run);
        writetable(sortrows(resp_map,"stim_id"),fullfile(output_dir,ids{isub},resp_map_fn))
        
        %hierachy model: stimuli location wrt screen centre
        olddata.x_sign = sign(olddata.stim_x);
        olddata.x_dist = abs(olddata.stim_x);
        olddata.y_sign = sign(olddata.stim_y);
        olddata.y_dist = abs(olddata.stim_y);


        for t = 1:size(olddata,1)
            % construct regressors for repetition suppression analysis
            % {'difffeature','diffcolor','diffshape','diffx','diffy','dist2deuc','distx','disty'}
            if t == 1 || olddata.ctrl_resp(t-1) % exclude from repetition suppression analysis if in trial 1 or if response is required in the previous trial
                % distance based on feature model
                olddata.difffeature(t)   = nan;
                olddata.diffcolor(t)     = nan;
                olddata.diffshape(t)     = nan;
                olddata.diffx(t)         = nan;
                olddata.diffy(t)         = nan;

                % distance based on 2d model (euclidean)
                olddata.dist2deuc(t)        = nan;
                olddata.distx(t)         = nan;
                olddata.disty(t)         = nan;

                olddata.rstrials(t)      = nan;
                olddata.excluders(t)     = 1; 
            else
                % distance based on 10d model (feature-based)
                olddata.difffeature(t)   = sum(abs([olddata.stim_attrx(t),olddata.stim_attry(t)]-[olddata.stim_attrx(t-1),olddata.stim_attry(t-1)])); % i.e. number of different features
                olddata.diffcolor(t)     = 1 - strcmp(olddata.stim_color(t),olddata.stim_color(t-1));
                olddata.diffshape(t)     = 1 - strcmp(olddata.stim_shape(t),olddata.stim_shape(t-1));
                olddata.diffx(t)         = 1 - (olddata.stim_attrx(t)==olddata.stim_attrx(t-1));
                olddata.diffy(t)         = 1 - (olddata.stim_attry(t)==olddata.stim_attry(t-1));
                
                % distance based on 2d model (groundtruth or response)
                olddata.dist2deuc(t)        = norm([olddata.stim_x(t),olddata.stim_y(t)]-[olddata.stim_x(t-1),olddata.stim_y(t-1)]);               
                olddata.distx(t)         = abs(olddata.stim_x(t)-olddata.stim_x(t-1));
                olddata.disty(t)         = abs(olddata.stim_y(t)-olddata.stim_y(t-1));

                resp_loc_curr     = [resp_map.resp_x(resp_map.stim_id==olddata.stim_id(t)),resp_map.resp_y(resp_map.stim_id==olddata.stim_id(t))];
                resp_loc_prev     = [resp_map.resp_x(resp_map.stim_id==olddata.stim_id(t-1)),resp_map.resp_y(resp_map.stim_id==olddata.stim_id(t-1))];                
                %olddata.dist2d_resp(t) =  norm(resp_loc_curr-resp_loc_prev);

                olddata.rstrials(t)    = 1;
                olddata.excluders(t)   = nan;

            end
            % construct regressors for train>test stimuli analysis as well
            % as RS separated by train vs test
            if olddata.training(t)
                olddata.onset_training(t)    = olddata.onset_stimuli(t);
                olddata.duration_training(t) = olddata.duration_stimuli(t);
                olddata.onset_test(t)    = nan;
                olddata.duration_test(t) = nan;
                
                if t==1 || olddata.training(t-1)==0 % if previoust trial is test
                    trainrs = nan;
                    crossrs = 1;
                else
                    trainrs = 1;
                    crossrs = nan;
                end

                olddata.onset_rstraining(t)  = olddata.onset_stimuli(t)*olddata.rstrials(t)*trainrs;
                olddata.onset_rstest(t)      = nan;
                olddata.onset_rstraintest(t) = olddata.onset_stimuli(t)*olddata.rstrials(t)*crossrs;
                olddata.duration_rstraining(t) = olddata.duration_stimuli(t)*olddata.rstrials(t)*trainrs;
                olddata.duration_rstest(t)     = nan;
                olddata.duration_rstraintest(t) = olddata.duration_stimuli(t)*olddata.rstrials(t)*crossrs;
            else
                olddata.onset_training(t)    = nan;
                olddata.duration_training(t) = nan;
                olddata.onset_test(t)    = olddata.onset_stimuli(t);
                olddata.duration_test(t) = olddata.duration_stimuli(t);

                if t==1 || olddata.training(t-1)==1 % if previoust trial is train
                    testrs = nan;
                    crossrs = 1;
                else
                    testrs = 1;
                    crossrs = nan;
                end

                olddata.onset_rstraining(t)    = nan;
                olddata.onset_rstest(t)        = olddata.onset_stimuli(t)*olddata.rstrials(t)*testrs;
                olddata.onset_rstraintest(t) = olddata.onset_stimuli(t)*olddata.rstrials(t)*crossrs;
                olddata.duration_rstraining(t) = nan;
                olddata.duration_rstest(t)     = olddata.duration_stimuli(t)*olddata.rstrials(t)*testrs;
                olddata.duration_rstraintest(t) = olddata.duration_stimuli(t)*olddata.rstrials(t)*crossrs;
            end
            % assign 'resp' location to current stimulus even if no response is
            % required for current trial for neural axis glm analysis
            olddata.respmap_x(t)    = resp_map.resp_x(resp_map.stim_id==olddata.stim_id(t));
            olddata.respmap_y(t)    = resp_map.resp_y(resp_map.stim_id==olddata.stim_id(t));
        end
        olddata.onset_excluders    = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials     = olddata.rstrials .* olddata.onset_stimuli;
        olddata.duration_excluders = olddata.excluders .* olddata.duration_stimuli;
        olddata.duration_rstrials  = olddata.rstrials .* olddata.duration_stimuli;
        
        ctrl_resp = olddata.ctrl_resp;
        ctrl_resp(ctrl_resp==0) = nan;
        olddata.onset_training_resp    = olddata.onset_training.*ctrl_resp;
        olddata.duration_training_resp = olddata.duration_training.*ctrl_resp;
        olddata.onset_test_resp    = olddata.onset_test.*ctrl_resp;
        olddata.duration_test_resp = olddata.duration_test.*ctrl_resp;
        
        ctrl_noresp = 1-olddata.ctrl_resp;
        ctrl_noresp(ctrl_noresp==0) = nan;
        olddata.onset_training_noresp    = olddata.onset_training.*ctrl_noresp;
        olddata.duration_training_noresp = olddata.duration_training.*ctrl_noresp;
        olddata.onset_test_noresp    = olddata.onset_test.*ctrl_noresp;
        olddata.duration_test_noresp = olddata.duration_test.*ctrl_noresp;
        
        % construct regressors for each stimuli as an individual event
        unique_ids = unique(olddata.stim_id);
        for j = 1:numel(unique_ids)
            sid = unique_ids(j);
            olddata.(sprintf('onset_stim%02d',sid)) = (olddata.stim_id==sid) .* olddata.onset_stimuli;
            olddata.(sprintf('onset_stim%02d',sid))(olddata.stim_id~=sid) = nan;
            olddata.(sprintf('duration_stim%02d',sid)) = (olddata.stim_id==sid) .* olddata.duration_stimuli;
            olddata.(sprintf('duration_stim%02d',sid))(olddata.stim_id~=sid) = nan;

            olddata.(sprintf('onset_stim%02dresp',sid)) = olddata.(sprintf('onset_stim%02d',sid)) .* ctrl_resp;
            olddata.(sprintf('duration_stim%02dresp',sid)) = olddata.(sprintf('duration_stim%02d',sid)) .* ctrl_resp;

            olddata.(sprintf('onset_stim%02dnoresp',sid)) = olddata.(sprintf('onset_stim%02d',sid)) .* ctrl_noresp;
            olddata.(sprintf('duration_stim%02dnoresp',sid)) = olddata.(sprintf('duration_stim%02d',sid)) .* ctrl_noresp;

        end
        olddata.stim_attrx = olddata.stim_attrx+1;
        olddata.stim_attry = olddata.stim_attry+1;

        % devide trial into before resp and after resp
        firstresptrials = arrayfun(@(x) find((olddata.ctrl_resp).*(olddata.stim_id==x)),unique(olddata.stim_id));
        is_before_firstresp = @(trial_no) 1*(trial_no<=firstresptrials(olddata.stim_id(trial_no)+1));
        beforefirstresp = arrayfun(is_before_firstresp,1:size(olddata,1))';
        
        ctrl_bresp = beforefirstresp;
        ctrl_bresp(ctrl_bresp==0) = nan;
        olddata.onset_training_beforeresp    = olddata.onset_training.*ctrl_bresp;
        olddata.duration_training_beforeresp = olddata.duration_training.*ctrl_bresp;
        olddata.onset_test_beforeresp    = olddata.onset_test.*ctrl_bresp;
        olddata.duration_test_beforeresp = olddata.duration_test.*ctrl_bresp;
        
        ctrl_aresp = 1-beforefirstresp;
        ctrl_aresp(ctrl_aresp==0) = nan;
        olddata.onset_training_afterresp    = olddata.onset_training.*ctrl_aresp;
        olddata.duration_training_afterresp = olddata.duration_training.*ctrl_aresp;
        olddata.onset_test_afterresp    = olddata.onset_test.*ctrl_aresp;
        olddata.duration_test_afterresp = olddata.duration_test.*ctrl_aresp;
        unique_ids = unique(olddata.stim_id);
        for j = 1:numel(unique_ids)
            sid = unique_ids(j);
            olddata.(sprintf('onset_stim%02dbefore',sid)) = olddata.(sprintf('onset_stim%02d',sid)) .* ctrl_bresp;
            olddata.(sprintf('duration_stim%02dbefore',sid)) = olddata.(sprintf('duration_stim%02d',sid)) .* ctrl_bresp;
            olddata.(sprintf('onset_stim%02dafter',sid)) = olddata.(sprintf('onset_stim%02d',sid)) .* ctrl_aresp;
            olddata.(sprintf('duration_stim%02dafter',sid)) = olddata.(sprintf('duration_stim%02d',sid)) .* ctrl_aresp;
        end

        keep_v = {'stim_id','stim_img','stim_x','stim_y','stim_attrx','stim_attry','resp_x','resp_y',...% fields in the orginal data table
                  'respmap_x','respmap_y',...% response map 
                  'onset_stimuli','duration_stimuli',... % stimuli event
                  'onset_response','duration_response',... % response event                
                  'onset_rstrials','duration_rstrials',...  % repetition suppression event
                  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
                  'difffeature','diffcolor','diffshape','diffx','diffy','dist2deuc','distx','disty',...% RS parametric modulators for distance
                  'onset_training','duration_training','onset_rstraining','duration_rstraining',... % training event   
                  'onset_test','duration_test','onset_rstest','duration_rstest','onset_rstraintest','duration_rstraintest',... % test event 
                  'onset_training_resp','duration_training_resp','onset_test_resp','duration_test_resp',...
                  'onset_training_noresp','duration_training_noresp','onset_test_noresp','duration_test_noresp',...
                  'onset_training_beforeresp','duration_training_beforeresp','onset_test_beforeresp','duration_test_beforeresp',...
                  'onset_training_afterresp','duration_training_afterresp','onset_test_afterresp','duration_test_afterresp'...
                };
        keep_v = [keep_v,...
                  arrayfun(@(x) sprintf('onset_stim%02d',x),unique_ids,'uni',0)',...% separated stimulus event onset
                  arrayfun(@(x) sprintf('duration_stim%02d',x),unique_ids,'uni',0)',...%#ok<*AGROW> % separated stimulus event duration
                  arrayfun(@(x) sprintf('onset_stim%02dresp',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('duration_stim%02dresp',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('onset_stim%02dnoresp',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('duration_stim%02dnoresp',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('onset_stim%02dbefore',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('duration_stim%02dbefore',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('onset_stim%02dafter',x),unique_ids,'uni',0)',...
                  arrayfun(@(x) sprintf('duration_stim%02dafter',x),unique_ids,'uni',0)']; 
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-piratenavigation_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        save(fullfile(output_dir,ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-piratenavigation_run-%d.mat',...
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
        
        olddata.onset_stimuli    = olddata.time_cue - meta.time.task_start;
        for t = 1:size(olddata,1)
            if isnan(olddata.time_response(t))
                olddata.duration_stimuli(t) = olddata.time_jitter(t) - olddata.time_cue(t);
            else
                olddata.duration_stimuli(t) = olddata.time_response(t) - olddata.time_cue(t);
            end
        end
        olddata.onset_response = olddata.time_response - meta.time.task_start;% no duration of response, will be set as stick function in glm
        olddata.response = olddata.resp_same;
        % replace nan with no go (0) except for the first trial
        tmp              = olddata.resp_same(2:end);
        tmp(isnan(tmp))  = 0;
        olddata.acc      = [nan;tmp == olddata.stim_same1back(2:end)];

        % construct regressors for repetition suppression analysis
        for t = 1:size(olddata,1)
            % {'diffcolor','diffshape','diffx','diffy','dist2deuc','distx','disty'}
            if t == 1 % exclude from repetition suppression analysis if in trial 1
                % distance based on feature model   
                olddata.diffx(t)         = nan;
                olddata.diffy(t)         = nan;

                % distance based on 2d model (euclidean)
                olddata.dist2deuc(t)        = nan;
                olddata.distx(t)         = nan;
                olddata.disty(t)         = nan;

                olddata.rstrials(t)      = nan;
                olddata.excluders(t)     = 1; 

                olddata.prev_stimx(t) = nan;
                olddata.prev_stimy(t) = nan;
            else
                % distance based on 10d model (feature-based)
                olddata.diffx(t)         = 1 - (olddata.stim_attrx(t)==olddata.stim_attrx(t-1));
                olddata.diffy(t)         = 1 - (olddata.stim_attry(t)==olddata.stim_attry(t-1));
                
                % distance based on 2d model (groundtruth or response)
                olddata.dist2deuc(t)     = norm([olddata.stim_x(t),olddata.stim_y(t)]-[olddata.stim_x(t-1),olddata.stim_y(t-1)]);               
                olddata.distx(t)         = abs(olddata.stim_x(t)-olddata.stim_x(t-1));
                olddata.disty(t)         = abs(olddata.stim_y(t)-olddata.stim_y(t-1));

                olddata.rstrials(t)    = 1;
                olddata.excluders(t)   = nan;

                olddata.prev_stimx(t) = olddata.stim_x(t-1);
                olddata.prev_stimy(t) = olddata.stim_y(t-1);
            end
        end
        olddata.onset_excluders = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials  = olddata.rstrials .* olddata.onset_stimuli;
        olddata.duration_excluders = olddata.excluders .* olddata.duration_stimuli;
        olddata.duration_rstrials  = olddata.rstrials .* olddata.duration_stimuli;
        
        unique_ids = unique(olddata.stim_id);
        for j = 1:numel(unique_ids)
            sid = unique_ids(j);
            olddata.(sprintf('onset_stim%02d',sid)) = (olddata.stim_id==sid) .* olddata.onset_stimuli;
            olddata.(sprintf('onset_stim%02d',sid))(olddata.stim_id~=sid) = nan;
            olddata.(sprintf('duration_stim%02d',sid)) = (olddata.stim_id==sid) .* olddata.duration_stimuli;
            olddata.(sprintf('duration_stim%02d',sid))(olddata.stim_id~=sid) = nan;
        end
        olddata.stim_attrx = olddata.stim_attrx+1;
        olddata.stim_attry = olddata.stim_attry+1;

        keep_v = {'stim_id','stim_img','stim_x','stim_y','stim_attrx','stim_attry','response','acc',...% fields in the orginal data table
                  'prev_stimx','prev_stimy',...
                  'excluders','rstrials',...
                  'onset_stimuli',  'duration_stimuli',... % stimuli event
                  'onset_response',... % response event, no duration of response, will be set as stick function in glm                
                  'onset_rstrials', 'duration_rstrials',...  % repetition suppression event
                  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
                  'diffx','diffy','dist2deuc','distx','disty'
                 };
        keep_v = [keep_v,...
                  arrayfun(@(x) sprintf('onset_stim%02d',x),unique_ids,'uni',0)',...#
                  arrayfun(@(x) sprintf('duration_stim%02d',x),unique_ids,'uni',0)'];              
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-localizer_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        save(fullfile(output_dir,ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-localizer_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
        clearvars exptid bhav_data_pat meta_data_pat olddata data keep_v remove_v
    end
end

