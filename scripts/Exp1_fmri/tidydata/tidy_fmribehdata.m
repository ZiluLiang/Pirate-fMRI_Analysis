% this files organizes behavioral data during fmri scans into the the
% format for univariate first level analysis. renamed files are in the same convention as
% renaed fmri files, which is:
% sub-x_task-taskname_run-j.mat
% additionally, the same table is saved as a sub-x_task-taskname_run-j_onsets.csv 
% for gaze decoding analysis in python
% taskname is 'piratenavigation' or 'localizer', x is subid, j is run
% each mat file contains a table with onset_eventname and duration_eventname
% if an event did not happen in a given trial, the value of onset and duration
% in the corresponding row will be set to nan.
% 
% each participant's response map in each pirate navigation task run
% is saved in a sub-x_task-taskname_run-j.csv file

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
        % we are setting the start of response screen as the onset of the
        % continuous movement (if any), and duration is just how long until
        % submit the response
        olddata.onset_response    = olddata.time_arena - meta.time.task_start;
        olddata.duration_response = olddata.time_jitter - olddata.time_arena;
        % we additionally build a column of onset_submit, for gaze decoding
        % analysis, this is based on the time_response column in the
        % original data, we just align it to start of scan
        olddata.onset_submit      = olddata.time_response - meta.time.task_start;
        
        % recontruct participant's map of stimuli based on their response
        % and calculate pairwise distance between stimuli for the current
        % run
        resp_map = [olddata.stim_id, olddata.resp_x, olddata.resp_y];
        resp_map = array2table(resp_map(olddata.ctrl_resp==1,:),'VariableNames',{'stim_id','resp_x','resp_y'});
        resp_map_fn = sprintf('sub-%s_task-piratenavigation_run-%d.csv',...
                                strrep(ids{isub},'sub',''), run);
        writetable(sortrows(resp_map,"stim_id"),fullfile(output_dir,ids{isub},resp_map_fn), "WriteMode","overwrite")
        disp(size(resp_map))
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

        % make columns for moving direction - vector pointing from starting
        % location to end response location
        olddata.moving_x    = olddata.resp_x - olddata.start_x;
        olddata.moving_y    = olddata.resp_y - olddata.start_y;
        % compute the moving angle and the sin&cos
        olddata.moving_theta = atan2(olddata.moving_y, olddata.moving_x);
        olddata.sin_movtheta = sin(olddata.moving_theta);
        olddata.cos_movtheta = cos(olddata.moving_theta);

        olddata.moving_absx = abs(olddata.moving_x);
        olddata.moving_absy = abs(olddata.moving_y);
        olddata.moving_xdir = sign(olddata.moving_x);
        olddata.moving_ydir = sign(olddata.moving_y);
        % double check if response trials has moving direction specified
        mov_check = all(isnan(olddata.moving_x) == isnan(olddata.time_arena)) && all(isnan(olddata.moving_y) == isnan(olddata.time_arena));
        assert(mov_check);
        noresp_mask = isnan(olddata.time_arena);
        % construct four events that correspond to: 
        % x-moveleft, x-moveright, y-moveup, y-movedown
        % note that signy<0 is move up, signx<0 is moveleft
        olddata.moveleft = olddata.moving_xdir<0;
        olddata.moveright = olddata.moving_xdir>0;
        olddata.moveup = olddata.moving_ydir<0;
        olddata.movedown = olddata.moving_ydir>0;

        olddata{:, {'onset_moveleft','onset_moveright','onset_moveup','onset_movedown'}} = NaN;
        olddata{:, {'duration_moveleft','duration_moveright','duration_moveup','duration_movedown'}} = NaN;
        olddata{:, {'pathlen_moveleft','pathlen_moveright','pathlen_moveup','pathlen_movedown'}} = NaN;
        % onsets
        olddata.onset_moveleft(olddata.moveleft==1)   = olddata.onset_response(olddata.moveleft==1);
        olddata.onset_moveright(olddata.moveright==1) = olddata.onset_response(olddata.moveright==1);
        olddata.onset_moveup(olddata.moveup==1)       = olddata.onset_response(olddata.moveup==1);
        olddata.onset_movedown(olddata.movedown==1)   = olddata.onset_response(olddata.movedown==1);
        % duration
        olddata.duration_moveleft(olddata.moveleft==1)   = olddata.duration_response(olddata.moveleft==1);
        olddata.duration_moveright(olddata.moveright==1) = olddata.duration_response(olddata.moveright==1);
        olddata.duration_moveup(olddata.moveup==1)       = olddata.duration_response(olddata.moveup==1);
        olddata.duration_movedown(olddata.movedown==1)   = olddata.duration_response(olddata.movedown==1);
        % parametric modulator
        olddata.pathlen_moveleft(olddata.moveleft==1)   = olddata.moving_absx(olddata.moveleft==1);
        olddata.pathlen_moveright(olddata.moveright==1) = olddata.moving_absx(olddata.moveright==1);
        olddata.pathlen_moveup(olddata.moveup==1)       = olddata.moving_absy(olddata.moveup==1);
        olddata.pathlen_movedown(olddata.movedown==1)   = olddata.moving_absy(olddata.movedown==1);

        movdirs = {'left','right','up','down'};
        mov_dir_variables = [{'moving_x','moving_y','moving_theta','sin_movtheta','cos_movtheta'}, ...
        cellfun(@(x) sprintf('onset_move%s',x),movdirs,'UniformOutput',false), ...
        cellfun(@(x) sprintf('duration_move%s',x),movdirs,'UniformOutput',false), ...
        cellfun(@(x) sprintf('pathlen_move%s',x),movdirs,'UniformOutput',false), ...
        ];

        % take a look at the columns to double check
        %olddata(:, [{'onset_response','duration_response'},mov_dir_variables])


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

        keep_v = {'stim_id','stim_img','stim_x','stim_y','stim_attrx','stim_attry','start_x','start_y','resp_x','resp_y',...% fields in the orginal data table
                  'respmap_x','respmap_y',...% response map 
                  'onset_stimuli','duration_stimuli',... % stimuli event
                  'onset_response','duration_response','onset_submit',... % response event                
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
                  arrayfun(@(x) sprintf('duration_stim%02dafter',x),unique_ids,'uni',0)',...
                  mov_dir_variables]; 
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        new_filename = sprintf('sub-%s_task-piratenavigation_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        save(fullfile(output_dir,ids{isub},new_filename),'data')

        datacsv_fn = sprintf('sub-%s_task-piratenavigation_run-%d_onsets.csv', strrep(ids{isub},'sub',''), run);
        writetable(data,fullfile(output_dir,ids{isub},datacsv_fn))


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
        
        datacsv_fn = sprintf('sub-%s_task-localizer_run-%d_onsets.csv', strrep(ids{isub},'sub',''), run);
        writetable(data,fullfile(output_dir,ids{isub},datacsv_fn))

        new_meta_fn = sprintf('metadata_sub-%s_task-localizer_run-%d.mat',...
                                strrep(ids{isub},'sub',''), run);
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
        clearvars exptid bhav_data_pat meta_data_pat olddata data keep_v remove_v
    end
end

