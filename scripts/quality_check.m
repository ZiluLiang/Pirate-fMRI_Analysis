% quality checks of fmri data preprocessing
% TODO: checkout
% https://github.com/jsheunis/fMRwhy/tree/master/fmrwhy/qcand see if more
% quality control measures should be calculated
% ------ written by Zilu Liang(2023.5,Oxford)------

clear;clc
%% Configurations
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');                                                         
qc_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\qualitycheck';



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
        rand_run = randi(n_runs(task));
        for run = rand_run%1:n_runs(task)
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


%% --------------  Quality Check for head motion: line plot of rp parameters  -------------- 
% set up parallel pool
num_workers   = feature('NumCores');
poolobj       =  parpool(num_workers);%set up parallel processing
% run head motion check
hm_dir = fullfile(qc_dir,'headmotion');
hm_tables = cell(participants.nsub,1);
checkdir(hm_dir)
parfor isub = 1:participants.nsub
    fprintf('ploting head motion for %s\n',participants.ids{isub});
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    [hm_tables{isub},f] = check_head_motion(subimg_dir,[-5,4],2.5,false,true,false);
    participant = repmat(participants.ids(isub),size(hm_tables{isub},1),1);
    hm_tables{isub} = addvars(hm_tables{isub},participant,'Before',1);    
    saveas(f,fullfile(hm_dir,[participants.ids{isub},'.png']));
    close all     
end
hm_table = cat(1,hm_tables{:});
writetable(hm_table,fullfile(qc_dir,'QualityCheckLogBook.xlsx'),'Sheet','headmotion')
% close parallel pool
delete(poolobj)

%% --------------  Quality Check for head motion: view 4D images as animation in mricro GL  -------------- 
for isub = 1:participants.nsub
    fprintf('viewing 4d series for %s\n',participants.ids{isub});
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    check_head_motion(subimg_dir,[-5,4],2.5,true,false,false);
    pause
end

%% --------------  Calculate tSNR for each run of each participants  -------------- 
mask_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks';
masks_names = cellstr(spm_select('List',mask_dir,'.*.nii'));
masks = cell2struct(fullfile(mask_dir,masks_names),cellfun(@(x) strrep(x,'.nii',''),masks_names,'uni',0));
mean_tsnr = nan(numel(participants.ids)*5,numel(masks_names));
tsnr_dir  = fullfile(qc_dir,'tsnr');
checkdir(tsnr_dir)
for isub = 1:participants.nsub
    fprintf('calculating tsnr for %s\n',participants.ids{isub});
    subimg_dir  = fullfile(directory.preprocess,participants.ids{isub});
    func_imgs   = cellstr(spm_select('FPList',subimg_dir,filepattern.preprocess.normalise));
    for j = 1:numel(func_imgs)
        src_img  = func_imgs{j};
        [~,fn,~] = fileparts(src_img);
        
        outputfn = fullfile(tsnr_dir,['tsnr_',fn,'.nii']);
        tmp_tsnr   = calculate_snr(src_img,outputfn,masks);
        mean_tsnr((isub-1)*5+j,:) = table2array(struct2table(tmp_tsnr));
    end
end
mean_tsnr_T = array2table(mean_tsnr,'VariableNames',masks_names);
tmp = fullfact([5,numel(participants.ids)]);
mean_tsnr_T.Properties.RowNames = arrayfun(@(k) [participants.ids{tmp(k,2)},'-run',num2str(tmp(k,1))],1:size(tmp,1),'uni',0);
writetable(mean_tsnr_T,fullfile(qc_dir,'QualityCheckLogBook.xlsx'),'Sheet','tsnr')