% fmri data preprocessing
% The script contains seven steps controlled by seven flags in the preprocess_flags struct:
%       reorient:       auto-reorient T1/fmri/fieldmap according to a template
%       calVDM:         calculated voxel displacement map
%       realign_unwarp: realign and unwarp using vdm
%       coregistration: coregister t1 to mean epi
%       segmentation:   segment and normalise coregistered T1 according to template
%       normalisation:  use the estimated normalization parameters to normalise epi images to mni space
%       smooth:         spatial smoothing
% Quality inspection should be taken in between steps to check preprocessing quality
% 
% After preprocessing: nuisance regressors are created for conducting first
% level glm analysis, preprocessed images are copied to new folders for
% subsequent analysis
%
% -----------------------------------------------------------------------    
% Author: Zilu Liang

clear;clc
%% Configurations
[directory,participants,filepattern,fmri] = get_pirate_defaults(false,'directory','participants','filepattern','fmri');

% Configure the steps, field names of the preprocess_flags struct must not be changed 
preprocess_flags   = struct('reorient',       true,...
                            'calVDM',         true,...
                            'realign_unwarp', true,...
                            'coregistration', true,...
                            'segmentation',   true,...
                            'normalisation',  true,...
                            'smooth',         true);                       
generate_nuisance = true;
copy_preprocessed = true;


%% Function handles for preprocessing
preprocess_handles  = struct('reorient',       @reorient,...
                             'calVDM',         @calculateVDM,...
                             'realign_unwarp', @realign_unwarp,...
                             'coregistration', @coregister,...
                             'segmentation',   @segment,...
                             'normalisation',  @normalise,...
                             'smooth',         @smooth);
                                             
%% set up parallel pool
if any([cell2mat(struct2cell(preprocess_flags))',generate_nuisance,copy_preprocessed])
    num_workers   = min([feature('NumCores')-4,nsub]);
    poolobj       =  parpool(num_workers);%set up parallel processing
    %create temporary variables so that we can minimize the amount of data sent to different parallel workers
    ids           = participants.validids;
    nsub          = participants.nvalidsub;
    preproc_dir   = directory.preprocess;
end                        
%% -----------------------  Preprocess data  ---------------------- 
preproc_steps = fieldnames(preprocess_handles);
preproc_steps = preproc_steps(cellfun(@(s) preprocess_flags.(s),preproc_steps)); % only run steps where flag is true

err_tracker   = cell(nsub,numel(preproc_steps));%initialize error tracker to record errors
for j = 1:numel(preproc_steps)
    curr_step   = preproc_steps{j};
    curr_handle = preprocess_handles.(curr_step);
    fprintf('Running %s\n\n', curr_step)
    
    for isub = 1:nsub
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

%% --------------  Generate Nuisance Regressor for head motion  -------------- 
% gen nuisance regressor using head motion parameters and their first derivatives
if generate_nuisance
    err_tracker2 = cell(participants.nsub,1);
    parfor isub = 1:nsub
        fprintf('creating head motion regressor for %s\n',ids{isub});
        try
            subimg_dir  = fullfile(directory.preprocess,ids{isub});
            generate_nuisance_regressor(subimg_dir); 
        catch err
            fprintf('Error generating nuisance regressor for %s %d/%d subject\n', ids{isub}, isub, nsub)
            err_tracker2{isub} = err;
        end
    end
end


%% -----------------  Copy Files: UnSmoothed -------------------
% After preprocessing is finished, create a copy of preprocessed files in a
% clean folder for subsequent statistical analysis
if copy_preprocessed
    par_dir    = directory.unsmoothed; %#ok<*UNRCH>
    move_files = {filepattern.preprocess.normalise,filepattern.preprocess.nuisance};
    parfor isub  = 1:nsub
        from_dir = fullfile(preproc_dir,ids{isub});
        to_dir   = fullfile(par_dir,ids{isub});
        checkdir(to_dir)
        src_fns  = cellfun(@(p) cellstr(spm_select('FPList',fullfile(preproc_dir,ids{isub}),p)),move_files,'uni',0)
        fprintf('Copying %s %d/%d subject\n', ids{isub}, isub, nsub)
        cellfun(@(s) copyfile(s,to_dir),cat(1,src_fns{:}));
        fprintf('Completed copying UnSmoothed data %s %d/%d subject\n', ids{isub}, isub, nsub)
    end
    fprintf('Completed UnSmoothed data copying\n\n')
end

%% -----------------  Copy Files: Smoothed -------------------
% After preprocessing is finished, create a copy of preprocessed files in a
% clean folder for subsequent statistical analysis
if copy_preprocessed
    par_dir    = directory.smoothed; %#ok<*UNRCH>
    move_files = {filepattern.preprocess.smooth,filepattern.preprocess.nuisance};
    parfor isub  = 1:nsub
        from_dir = fullfile(preproc_dir,ids{isub});
        to_dir   = fullfile(par_dir,ids{isub});
        checkdir(to_dir)
        src_fns  = cellfun(@(p) cellstr(spm_select('FPList',fullfile(preproc_dir,ids{isub}),p)),move_files,'uni',0)
        fprintf('Copying %s %d/%d subject\n', ids{isub}, isub, nsub)
        cellfun(@(s) copyfile(s,to_dir),cat(1,src_fns{:}));
        fprintf('Completed copying Smoothed data %s %d/%d subject\n', ids{isub}, isub, nsub)
    end
    fprintf('Completed Smoothed data copying\n\n')
end

%% close parallel pool
if any([cell2mat(struct2cell(preprocess_flags))',generate_nuisance,copy_preprocessed])
    delete(poolobj)
end