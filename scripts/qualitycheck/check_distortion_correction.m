function check_distortion_correction(subimg_dir,varargin)
% show the unwarped epi, original epi and t1 image side by side to check
% the quality of distortion correction using fieldmap. 
% example usage: checkDistortionCorrection(subimg_dir,'task2',1,400)

    if numel(varargin)>3
        error('Too many input arguments')
    end
    
    % get the filepattern from default setting 
    filepattern = get_pirate_defaults(false,'filepattern');
    
    % specify which task,which run, which volume to check,
    % select the first volume of the first session(1st run of 1st task) by default
    task_run_volume = {1,1,1};
    task_run_volume(1:numel(varargin)) = varargin(:);
    [task,run,volume] = task_run_volume{:};
    taskname = ['task',num2str(task)];
    
    % look for fieldmap corrected epi. if realign_unwarped is not yet
    % performed, find the warped first image of first session from vdm
    % calculation step if 1st volume of 1st run of 1st task is queried
    unwarpedepi = spm_select('FPList',subimg_dir,[filepattern.preprocess.realignunwarp,'.*',strrep(filepattern.raw.functional.(taskname),'^',''),'*.nii']);
    if isempty(unwarpedepi)
        if all([task,run,volume] == 1)
            unwarpedepi = spm_select('FPList',subimg_dir,filepattern.preprocess.firstepiunwarp);
        end
    end
    if isempty(unwarpedepi)
        error('fieldmap corrected epi image not found!')
    else
        unwarpedepi = cellstr(unwarpedepi);
    end
    
    if volume>size(spm_vol(unwarpedepi{run}),1)
        error('specified volume exceed total number of volumes!')
    end
    unwarpedepi = cellstr([unwarpedepi{run},',',num2str(volume)]);
    
    %If reorientation is performed before distortion correction, diplay
    %reoriented images to compare
    [~,unwarpedepi_fn,~] = fileparts(unwarpedepi);
    startind = regexp(unwarpedepi_fn,strrep(filepattern.reorient.functional.(taskname),'^',''), 'once');
    if isempty(startind)
       originalepi = cellstr(spm_select('FPList',subimg_dir,[filepattern.raw.functional.(taskname),'.nii']));
       t1          = cellstr(spm_select('FPList',subimg_dir,[filepattern.raw.anatomical.T1,'.nii']));             

    else
       originalepi = cellstr(spm_select('FPList',subimg_dir,[filepattern.reorient.functional.(taskname),'.nii']));
       t1          = cellstr(spm_select('FPList',subimg_dir,[filepattern.reorient.anatomical.T1,'.nii']));             
    end    
    originalepi = cellstr([originalepi{run},',',num2str(volume)]);
        
    captions = {'unwarped epi','original epi','t1'};
    images   = [unwarpedepi,originalepi,t1];
    ext_spm_checkreg(images,captions)
end
