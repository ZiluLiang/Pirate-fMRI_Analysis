[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
beh_dir = directory.fmribehavior;
nii_dir = fullfile(directory.fmri_data,'renamed');

% pirate task run
for isub = 1:participants.nvalidsub
    subid      = participants.validids{isub};
    subimg_dir = fullfile(nii_dir,subid);
    for j = 1:4
        if isub ~= 9 && isub ~=24%these two participants has to be excluded because data files were modified and timing is not accurate
            % load json file with the header info of the scans
            nii_info_fn   = sprintf('%s_task-piratenavigation_run-%d.json',strrep(subid,'sub','sub-'),j);
            scanning_info = loadjson(spm_select('FPList',subimg_dir,nii_info_fn));
            % load mat file with the time info of the task
            beh_info_fn   = sprintf('metadata_%s_task-piratenavigation_run-%d.mat',strrep(subid,'sub','sub-'),j);
            load(spm_select('FPList',fullfile(beh_dir,'metadata'),beh_info_fn),'meta');
            
            % get time stamp of the first scan on the scanner computer, set
            % the format to six digits after seconds to improve precision
            scaner_start  = datetime(scanning_info.AcquisitionTime,'InputFormat','HH:mm:ss.SSSSSS','Format','yyyy-MM-dd HH:mm:ss.SSSSSS');

            % get time stamp of the data saving on the test computer, set
            % the format to six digits after seconds to improve precision
            % use data saving time as proxi for task end time
            task_save     = strsplit(dir(spm_select('FPList',fullfile(beh_dir,'metadata'),beh_info_fn)).date,' ');
            task_save     = datetime(task_save{2},'InputFormat','HH:mm:ss','Format','yyyy-MM-dd HH:mm:ss.SSSSSS');
            
            % estimate starting time of the task on the test computer
            task_duration = datenum(seconds(meta.time.task_end - meta.time.task_start));
            task_start    = task_save - task_duration;
            
            % log the time difference between start time on the test pc and
            % scanner pc
            pc_diff(isub,j) = datetime(seconds(scaner_start - task_start),'ConvertFrom','posixtime','Format','HH:mm:ss.SSSSSS');
        end
    end
end

% localizer task run
for isub = 1:participants.nvalidsub
    subid      = participants.validids{isub};
    subimg_dir = fullfile(nii_dir,subid);
    if isub ~= 9 && isub ~=24
        nii_info_fn   = [strrep(subid,'sub','sub-'),'_task-localizer_run-1.json'];
        scanning_info = loadjson(spm_select('FPList',subimg_dir,nii_info_fn));
        beh_info_fn   = ['metadata_',strrep(subid,'sub','sub-'),'_task-localizer_run-1.mat'];
        load(spm_select('FPList',fullfile(beh_dir,'metadata'),beh_info_fn),'meta');

        scaner_start  = datetime(scanning_info.AcquisitionTime,'InputFormat','HH:mm:ss.SSSSSS','Format','yyyy-MM-dd HH:mm:ss.SSSSSS');
        task_save     = strsplit(dir(spm_select('FPList',fullfile(beh_dir,'metadata'),beh_info_fn)).date,' ');
        task_save     = datetime(task_save{2},'InputFormat','HH:mm:ss','Format','yyyy-MM-dd HH:mm:ss.SSSSSS');
        
        task_duration = datenum(seconds(meta.time.task_end - meta.time.task_start));
        task_start    = task_save - task_duration;

        pc_diff(isub,5) = datetime(seconds(scaner_start - task_start),'ConvertFrom','posixtime','Format','HH:mm:ss.SSSSSS');
    end
end
pc_diff_T = array2table(pc_diff,'VariableNames',{'pirate-run1','pirate-run2','pirate-run3','pirate-run4','localizer'});
pc_diff_T.subid = participants.validids;
save(fullfile(directory.fmri_data,'timingchecks.mat'),'pc_diff_T');