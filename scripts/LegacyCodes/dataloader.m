function regressors = dataloader
    regressors = struct( );
    bhavdata_dir = 'D:\Dropbox\zilu\task\data_jirko';
    for irun = 1:4
        
        load(spm_select('FPListRec',bhavdata_dir,sprintf("^maintask_ILxuzJ19k33q_run%d.*.mat",irun)),'data')
        load(spm_select('FPListRec',bhavdata_dir,sprintf("^meta_maintask_ILxuzJ19k33q_run%d.*.mat",irun)),'meta')
        onset.cue = data.time_cue - meta.time.task_start;
        onset.arena = data.time_arena - meta.time.task_start;
        onset.response = data.time_response - meta.time.task_start;
        onset.jitter = data.time_jitter - meta.time.task_start;
        rt = data.time_response - data.time_arena;
        ntrial = size(data,1);
        % filter out nan
        onset_removenan = structfun(@(x) x(~isnan(x)),onset,'UniformOutput',false);
        regressors(irun).cue = struct('onset',{onset_removenan.cue},...
                           'duration',{ones(size(onset_removenan.cue))});
        regressors(irun).response = struct('onset',{onset_removenan.arena},...
                          'duration', {rt(~isnan(rt))});
    end

    clear("data")
    irun = 1;
    load(spm_select('FPListRec',bhavdata_dir,sprintf("^localizer_ILxuzJ19k33q_run%d.*.mat",irun)),'data')
    load(spm_select('FPListRec',bhavdata_dir,sprintf("^meta_localizer_ILxuzJ19k33q_run%d.*.mat",irun)),'meta')
    onset = struct();
    onset.cue = data.time_cue - meta.time.task_start;
    onset.response = data.time_response - meta.time.task_start;
    onset.jitter = data.time_jitter - meta.time.task_start;
    rt = data.time_response - data.time_cue;
    ntrial = size(data,1);
    % filter out nan
    regressors = struct();
    onset_removenan = structfun(@(x) x(~isnan(x)),onset,'UniformOutput',false);
    regressors(1).cue = struct('onset',{onset_removenan.cue},...
                               'duration',{zeros(size(onset_removenan.cue))});
    regressors(1).response = struct('onset',{onset_removenan.response},...
                                    'duration', {zeros(size(onset_removenan.response))});
end