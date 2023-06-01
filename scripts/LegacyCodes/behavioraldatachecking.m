load(fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\task\parameters','turker2id.mat'),'turker2id')
data_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\fmri\data\behavior';
turkers = fieldnames(turker2id);
turkers = turkers(cellfun(@(x) ~contains(x,'zilu'),fieldnames(turker2id)));
tasknames = {'maintask','localizer'};
nruns = [4,1];

compare_numarr = @(a,b) all([reshape(ismember(a,b),1,numel(ismember(a,b))),...
                             reshape(ismember(b,a),1,numel(ismember(b,a)))]);
for k = 1:numel(tasknames)
    taskname = tasknames{k};
    fprintf('=========== checking fmri data for task %s ===========\n', taskname)
    for j = 1:numel(turkers)
        for irun = 1:nruns(k)
           fprintf('participant %s run %d\n',turkers{j},irun)
           % loading data
           behav_fn = spm_select('FPList',data_dir,sprintf('^%s_%s_run%d.*.mat',taskname,turker2id.(turkers{j}),irun));
           load(behav_fn,'data');
           meta_fn = spm_select('FPList',data_dir,sprintf('^meta_%s_%s_run%d.*.mat',taskname,turker2id.(turkers{j}),irun));
           load(meta_fn,'meta');

           % checking if all stimuli is tested and only once in each run
           testedstim = data.stim_id(data.ctrl_resp == 1);
           showedstim = data.stim_id;
           if strcmp(taskname,'maintask')
               shouldbetestedstim = 0:1:24;
               fprintf('all test stimuli are shown 3 times: %d\n',all(arrayfun(@(x) sum(showedstim==x)==3,shouldbetestedstim)))
               fprintf('all test stimuli are tested only once: %d\n',compare_numarr(testedstim,shouldbetestedstim))
           else
               shouldbetestedstim = [2,7,12,17,22,11,13,14,15];
               all(arrayfun(@(x) sum(testedstim==x)==4,shouldbetestedstim))
               fprintf('all test stimuli are shown 10 times: %d\n',all(arrayfun(@(x) sum(showedstim==x)==10,shouldbetestedstim)))           
           end

           % checking timing
           switch taskname
               case 'maintask'
                   cue_offsettime = [data.time_jitter,data.time_arena];                   
                   cue = arrayfun(@(x) cue_offsettime(x,data.ctrl_resp(x)+1),1:size(cue_offsettime,1))' - data.time_cue;
                   jitter = [data.time_cue(2:end) - data.time_jitter(1:end-1);...
                                        meta.time.task_end - data.time_jitter(end)];
                   
               case 'localizer'
                   cue = data.time_jitter - data.time_cue;
                   jitter = [data.time_cue(2:end) - data.time_jitter(1:end-1);...
                                        meta.time.task_end - data.time_jitter(end)];
           end
           actualtime = table(cue,jitter);
           openvar("actualtime")
           keyboard
        end    
    end
end