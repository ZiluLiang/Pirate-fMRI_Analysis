% fmri data preprocessing: convert, rename to annonymized files and organize
% The script contains six steps controlled by six flags in the preprocess_flags struct:
%       calVDM:         calculated voxel displacement map
%       realign_unwarp: realign and unwarp using vdm
%       coregistration: coregister t1 to mean epi
%       segmentation:   segment and normalise coregistered T1 according to template
%       normalisation:  use the estimated normalization parameters to normalise epi images to mni space
%       smooth:         spatial smoothing
% Quality inspection should be taken in between steps to check preprocessing quality
%
% ------ written by Zillu Liang(2023.4,Oxford)------

clear;clc
%% Configurations
[directory,participants,filepattern,handles] = get_pirate_defaults(false,'directory','participants','filepattern','handles');

% Configure the steps, field names of the preprocess_flags struct must not be changed 
preprocess_flags   = struct('reorient',       false,...
                            'calVDM',         true,...
                            'realign_unwarp', true,...
                            'coregistration', true,...
                            'segmentation',   true,...
                            'normalisation',  true,...
                            'smooth',         false);                       

                        
%% -----------------------  Preprocess data  ---------------------- 
preproc_steps = fieldnames(handles.preprocess);
preproc_steps = preproc_steps(cellfun(@(s) preprocess_flags.(s),preproc_steps)); % only run steps where flag is true

num_workers   = feature('NumCores') - 1;
poolobj       =  parpool(num_workers);%set up parallel processing
%create temporary variables so that we can minimize the amount of data sent to different parallel workers
ids           = participants.ids;
nsub          = participants.nsub;
preproc_dir   = directory.preprocess;
%initialize error tracker to record errors
err_tracker   = cell(nsub,numel(preproc_steps));

for j = 1:numel(preproc_steps)
    curr_step   = preproc_steps{j};
    curr_handle = handles.preprocess.(curr_step);
    fprintf('Running %s\n\n', curr_step)
    
    parfor isub = 1:nsub
        fprintf('Running %s %d/%d subject\n', ids{isub}, isub, nsub)
        try % use try catch to minimize the chance of interrupted execution if error occurs in one of the jobs
            curr_handle(fullfile(preproc_dir,ids{isub}));  %#ok<*PFBNS>
            fprintf('Completed %s %d/%d subject\n', ids{isub}, isub, nsub)
        catch err
            fprintf('Error running %s for %s %d/%d subject\n', curr_step, ids{isub}, isub, nsub)
            err_tracker{isub,j} = err;
        end
    end
    fprintf('Completed %s\n\n', curr_step)
end
delete(poolobj)


%% -----------------  Copy Files -------------------
% After preprocessing is finished, create a copy of preprocessed files in a
% clean folder for subsequent statistical analysis
copy_preprocessed = false;
if copy_preprocessed
    par_dir    = directory.unsmoothed; %#ok<*UNRCH>
    move_files = {filepattern.preprocess.normalise,...
                  filepattern.preprocess.motionparam};
    parfor isub  = 1:nsub
        from_dir = fullfile(preproc_dir,ids{isub});
        to_dir   = fullfile(par_dir,ids{isub});
        checkdir(to_dir)
        src_fns  = cellfun(@(p) cellstr(spm_select('FPList',fullfile(preproc_dir,ids{isub}),p)),move_files,'uni',0)
        cellfun(@(s) copyfile(s,to_dir),cat(1,src_fns{:}));
    end
end