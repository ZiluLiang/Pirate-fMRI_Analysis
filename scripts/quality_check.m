% quality checks of fmri data preprocessing
%
% ------ written by Zillu Liang(2023.5,Oxford)------

clear;clc
%% Configurations
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');                                                         



%% --------------  Quality Check after reorientation  -------------- 
%check the if images are better aligned with template after auto
%reorientation
for isub = 1:participants.nsub
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    check_spatial_registration('all2template',subimg_dir)
    pause   
end



%% --------------  Quality Check after vdm calculation  -------------- 
%check the output of vdm that unwarp the first epi using the calculated vdm
for isub = 1:participants.nsub
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    check_distortion_correction(subimg_dir)
    pause
end


%% --------------  Quality Check after realign and unwarp  -------------- 
%select a random volume per participant per task per run just to quickly
%check signal loss and if t1 and functionals are aligned
n_tasks = 2;
n_runs  = [4,1];
n_volumes = {[296,296,296,296],[326]};
for isub = 1:participants.nsub
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    for task = 1:n_tasks
        for run = 1:n_runs(task)
            vol = randi(n_volumes{task}(run));
            check_distortion_correction(subimg_dir,task,run,vol);
            fprintf('showing %s-task%d-run%d-vol-%d',participants.ids{isub},task,run,vol)
            pause
        end        
    end    
end


%% --------------  Quality Check after coregistration  -------------- 
%overlay the mean epi on anatomical to see if there are displacement
for isub = 1:participants.nsub
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    check_spatial_registration('anat2epi',subimg_dir);
    pause   
end


%% --------------  Quality Check after normalization  -------------- 
%overlay the mean/first epi on anatomical to see if there are displacement
for isub = 1:participants.nsub
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    check_spatial_registration('epi2template',subimg_dir);
    pause   
end


%% --------------  Quality Check  for head motion  -------------- 
qc_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri_image\qualitycheck';
hm_dir = fullfile(qc_dir,'headmotion');
checkdir(hm_dir)
for isub = 1:participants.nsub
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    f = check_head_motion(subimg_dir,false,true,false);
    saveas(f,fullfile(hm_dir,[participants.ids{isub},'.png']));
    %pause   
    close all
end

