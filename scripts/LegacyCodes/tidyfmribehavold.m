% this files organizes behavioral data during fmri scans into the the
% format for glm processing. renamed files are in the same convention as
% renaed fmri files, which is:
% sub-x_task-taskname_run-j.mat
% where taskname is 'piratenavigation' or 'localizer', x is subid, j is run
% each mat file contains a table with onset_eventname and
% duration_eventname, if duraiont 


untidied_behaviordir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\untidiedbhavior\fmri_behavior';
renamer_fn = fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data','renamer.json');
renamer    = loadjson(renamer_fn);
participants = get_pirate_defaults(false,'participants');
output_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\beh';
checkdir(fullfile(output_dir,participants.ids))

%% navigation task
taskname = "maintask";
for isub = 1:participants.nsub
    fprintf('tidying navigation task %s\n',participants.ids{isub})
    for run = 1:4
        exptid = renamer.(participants.ids{isub}){1};
        bhav_data_pat = sprintf('^%s_%s_run%d_.*.mat',taskname,exptid,run);
        meta_data_pat = sprintf('^meta_%s_%s_run%d_.*.mat',taskname,exptid,run);
        load(spm_select('FPList',untidied_behaviordir,bhav_data_pat),'data');
        load(spm_select('FPList',untidied_behaviordir,meta_data_pat),'meta');
        olddata = data;
        clearvars data

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

        % construct regressors for repetition suppression analysis
        for t = 1:size(olddata,1)
            if t == 1 || isnan(olddata.time_arena(t-1)) % exclude from repetition suppression analysis if in trial 1 or if response is required in the previous trial
                olddata.samex(t) = nan;
                olddata.samey(t) = nan;
                olddata.diffx(t) = nan;
                olddata.diffy(t) = nan;
                olddata.dist2d(t) = nan;
                olddata.samecolor(t) = nan;
                olddata.sameshape(t) = nan;
                olddata.diffcolor(t) = nan;
                olddata.diffshape(t) = nan;
                olddata.excluders(t)  = 1; 
                olddata.rstrials(t)   = nan;                
            else
                features_t0 = strsplit(olddata.stim_img{t-1},'_');
                features_t  = strsplit(olddata.stim_img{t},'_');
                olddata.samex(t) = olddata.stim_x(t) == olddata.stim_x(t-1);
                olddata.samey(t) = olddata.stim_y(t) == olddata.stim_y(t-1);
                olddata.samecolor(t) = strcmp(features_t0{1},features_t{1});
                olddata.sameshape(t) = strcmp(features_t0{2},features_t{2});
                olddata.diffx(t) = ~olddata.samex(t);
                olddata.diffy(t) = ~olddata.samey(t);
                olddata.dist2d(t) =  norm([olddata.stim_x(t),olddata.stim_y(t)]-[olddata.stim_x(t-1),olddata.stim_y(t-1)]);
                olddata.diffcolor(t) = ~olddata.samecolor(t);
                olddata.diffshape(t) = ~olddata.sameshape(t);
                olddata.excluders(t) = nan;
                olddata.rstrials(t)   = 1;                
            end
            if olddata.training
                olddata.onset_training(t)    = olddata.onset_stimuli(t);
                olddata.duration_training(t) = olddata.duration_stimuli(t);
                olddata.onset_test(t)    = nan;
                olddata.duration_test(t) = nan;
            else
                olddata.onset_training(t)    = nan;
                olddata.duration_training(t) = nan;
                olddata.onset_test(t)    = olddata.onset_stimuli(t);
                olddata.duration_test(t) = olddata.duration_stimuli(t);
            end
        end
        olddata.onset_samex = olddata.samex .* olddata.onset_stimuli;
        olddata.onset_samey = olddata.samey .* olddata.onset_stimuli;
        olddata.onset_diffx = olddata.diffx .* olddata.onset_stimuli;
        olddata.onset_diffy = olddata.diffy .* olddata.onset_stimuli;
        olddata.onset_samecolor = olddata.samecolor .* olddata.onset_stimuli;
        olddata.onset_sameshape = olddata.sameshape .* olddata.onset_stimuli;
        olddata.onset_diffcolor = olddata.diffcolor .* olddata.onset_stimuli;
        olddata.onset_diffshape = olddata.diffshape .* olddata.onset_stimuli;
        olddata.onset_excluders = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials = olddata.rstrials .* olddata.onset_stimuli;

        keep_v = {'stim_id','stim_img','stim_x','stim_y',...
                  'samex','samey','samecolor','sameshape',...
                  'diffx','diffy','diffcolor','diffshape','dist2d','excluders','rstrials',...
                  'start_x','start_y','resp_x','resp_y','resp_dist',...
                  'onset_stimuli','duration_stimuli',...
                  'onset_response','duration_response',...
                  'onset_samex','onset_samey','onset_samecolor','onset_sameshape','onset_excluders',...
                  'onset_diffx','onset_diffy','onset_diffcolor','onset_diffshape','onset_rstrials',...
                  'onset_training','duration_training',...
                  'onset_test','duration_test',...
                };
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-piratenavigation_run-%d.mat',...
                                strrep(participants.ids{isub},'sub',''), run);
        save(fullfile(output_dir,participants.ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-piratenavigation_run-%d.mat',...
                                 strrep(participants.ids{isub},'sub',''), run);
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
    end
end

%% localizer task
taskname = "localizer";
for isub = 1:participants.nsub
    fprintf('tidying localizer task %s\n',participants.ids{isub})
    for run = 1
        clearvars data
        clearvars meta
        exptid = renamer.(participants.ids{isub}){1};
        bhav_data_pat = sprintf('^%s_%s_run%d_.*.mat',taskname,exptid,run);
        meta_data_pat = sprintf('^meta_%s_%s_run%d_.*.mat',taskname,exptid,run);
        load(spm_select('FPList',untidied_behaviordir,bhav_data_pat),'data');
        load(spm_select('FPList',untidied_behaviordir,meta_data_pat),'meta');
        olddata = data; 
        clearvars data

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
            if t == 1 % exclude from repetition suppression analysis if in trial 1
                olddata.samex(t) = nan;
                olddata.samey(t) = nan;
                olddata.diffx(t) = nan;
                olddata.diffy(t) = nan;
                olddata.dist2d(t) =  nan;
                olddata.excluders(t)  = 1;
                olddata.rstrials(t)   = nan;
            else
                olddata.samex(t) = olddata.stim_x(t) == olddata.stim_x(t-1);
                olddata.samey(t) = olddata.stim_y(t) == olddata.stim_y(t-1);
                olddata.diffx(t) = ~olddata.samex(t);
                olddata.diffy(t) = ~olddata.samey(t);
                olddata.dist2d(t) =  norm([olddata.stim_x(t),olddata.stim_y(t)]-[olddata.stim_x(t-1),olddata.stim_y(t-1)]);
                olddata.excluders(t)  = nan;
                olddata.rstrials(t)   = 1;
            end
        end
        olddata.onset_samex = olddata.samex .* olddata.onset_stimuli;
        olddata.onset_samey = olddata.samey .* olddata.onset_stimuli;
        olddata.onset_diffx = olddata.diffx .* olddata.onset_stimuli;
        olddata.onset_diffy = olddata.diffy .* olddata.onset_stimuli;
        olddata.onset_excluders = olddata.excluders .* olddata.onset_stimuli;
        olddata.onset_rstrials  = olddata.rstrials .* olddata.onset_stimuli;
        
        keep_v = {'stim_id','stim_img','stim_x','stim_y','samex','samey','diffx','diffy','dist2d','excluders','rstrials',...
                  'onset_stimuli','duration_stimuli',...
                  'onset_response',...
                  'onset_samex','onset_samey','onset_excluders','onset_rstrials',...
                  'onset_diffx','onset_diffy',...
                  'response','acc'...
                 };
        remove_v = olddata.Properties.VariableNames(cellfun(@(f) ~ismember(f,keep_v),olddata.Properties.VariableNames));
        data = removevars(olddata,remove_v);
        
        new_filename = sprintf('sub-%s_task-localizer_run-%d.mat',...
                                strrep(participants.ids{isub},'sub',''), run);
        save(fullfile(output_dir,participants.ids{isub},new_filename),'data')
        
        new_meta_fn = sprintf('metadata_sub-%s_task-localizer_run-%d.mat',...
                                strrep(participants.ids{isub},'sub',''), run);
        copyfile(spm_select('FPList',untidied_behaviordir,meta_data_pat),fullfile(output_dir,'metadata',new_meta_fn))
    end
end

