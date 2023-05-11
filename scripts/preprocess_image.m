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
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');

% Configure the steps, field names of the preprocess_flags struct must not be changed 
preprocess_flags   = struct('reorient',       false,...
                            'calVDM',         false,...
                            'realign_unwarp', false,...
                            'coregistration', false,...
                            'segmentation',   false,...
                            'normalisation',  true,...
                            'smooth',         false);                       
copy_preprocessed = false;


%% Function handles for preprocessing
preprocess_handles  = struct('reorient',       @reorient,...
                             'calVDM',         @calculateVDM,...
                             'realign_unwarp', @realign_unwarp,...
                             'coregistration', @coregister,...
                             'segmentation',   @segment,...
                             'normalisation',  @normalise,...
                             'smooth',         @smooth);
                                             
%% set up parallel pool
num_workers   = feature('NumCores') - 4;
poolobj       =  parpool(num_workers);%set up parallel processing
%create temporary variables so that we can minimize the amount of data sent to different parallel workers
ids           = participants.ids;
nsub          = participants.nsub;
preproc_dir   = directory.preprocess;
                        
%% -----------------------  Preprocess data  ---------------------- 
preproc_steps = fieldnames(preprocess_handles);
preproc_steps = preproc_steps(cellfun(@(s) preprocess_flags.(s),preproc_steps)); % only run steps where flag is true

err_tracker   = cell(nsub,numel(preproc_steps));%initialize error tracker to record errors
for j = 1:numel(preproc_steps)
    curr_step   = preproc_steps{j};
    curr_handle = preprocess_handles.(curr_step);
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


%% -----------------  Copy Files -------------------
% After preprocessing is finished, create a copy of preprocessed files in a
% clean folder for subsequent statistical analysis
if copy_preprocessed
    par_dir    = directory.unsmoothed; %#ok<*UNRCH>
    move_files = {filepattern.preprocess.normalise,...
                  filepattern.preprocess.motionparam};
    parfor isub  = 2:nsub
        from_dir = fullfile(preproc_dir,ids{isub});
        to_dir   = fullfile(par_dir,ids{isub});
        checkdir(to_dir)
        src_fns  = cellfun(@(p) cellstr(spm_select('FPList',fullfile(preproc_dir,ids{isub}),p)),move_files,'uni',0)
        fprintf('Copying %s %d/%d subject\n', ids{isub}, isub, nsub)
        cellfun(@(s) copyfile(s,to_dir),cat(1,src_fns{:}));
        fprintf('Completed copying %s %d/%d subject\n', ids{isub}, isub, nsub)
    end
    fprintf('Completed data copying\n\n')
end

%% close parallel pool
delete(poolobj)