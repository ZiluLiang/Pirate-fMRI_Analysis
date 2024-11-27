% quality checks of fmri data preprocessing
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO: checkout
% https://github.com/jsheunis/fMRwhy/tree/master/fmrwhy/qcand see if more
% quality control measures should be calculated

%% Configurations
clear;clc
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');                                                         
qc_dir = 'E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\qualitycheck';
ids  = participants.validids;
nsub = numel(ids);

%% --------------  Quality Check after reorientation  -------------- 
%check the if images are better aligned with template after auto
%reorientation
for isub = 1:nsub
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    check_spatial_registration('all2template',subimg_dir)
    pause   
end



%% --------------  Quality Check after vdm calculation  -------------- 
%check the output of vdm that unwarp the first epi using the calculated vdm
for isub = 1:nsub
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    check_distortion_correction(subimg_dir)
    pause
end


%% --------------  Quality Check after realign and unwarp  -------------- 
%select a random volume per participant per task per run just to quickly
%check signal loss and if t1 and functionals are aligned
n_tasks = 2;
n_runs  = [4,1];
n_volumes = {[296,296,296,296],[326]};
for isub = 1%:nsub
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    for task = 1:n_tasks
        rand_run = randi(n_runs(task));
        for run = rand_run%1:n_runs(task)
            vol = randi(n_volumes{task}(run));
            check_distortion_correction(subimg_dir,task,run,vol);
            fprintf('showing %s-task%d-run%d-vol-%d',ids{isub},task,run,vol)
            pause
        end        
    end    
end


%% --------------  Quality Check after coregistration  -------------- 
%overlay the mean epi on anatomical to see if there are displacement
for isub = 1:nsub
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    check_spatial_registration('anat2epi',subimg_dir);
    pause   
end


%% --------------  Quality Check after normalization  -------------- 
%overlay the mean/first epi on anatomical to see if there are displacement
for isub = 1:nsub
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    check_spatial_registration('epi2template',subimg_dir);
    pause   
end


%% --------------  Quality Check for head motion: line plot of rp parameters  -------------- 
% set up parallel pool
num_workers   = feature('NumCores');
poolobj       =  parpool(num_workers);%set up parallel processing
% run head motion check
hm_dir = fullfile(qc_dir,'headmotion');
hm_tables = cell(nsub,1);
checkdir(hm_dir)
rpthres = 2.5; % set half voxelsize as threshold
parfor isub = 1:nsub
    fprintf('ploting head motion for %s\n',ids{isub});
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    [hm_tables{isub},f] = check_head_motion(subimg_dir,[-5,4],rpthres,false,true,false);
    saveas(f,fullfile(hm_dir,[ids{isub},'.png']));
    
    participant = repmat(ids(isub),size(hm_tables{isub},1),1);
    taskname = {'maintask','maintask','maintask','maintask','localizer'}';
    taskrun  = [1:4,1]';
    hm_tables{isub} = addvars(hm_tables{isub},participant,taskname,taskrun,'Before',1,'NewVariableNames',{'subid','taskname','taskrun'}); 
    close all     
end
hm_table = cat(1,hm_tables{:});
writetable(hm_table,fullfile(qc_dir,'QualityCheckLogBook.xlsx'),'Sheet','headmotion_cohort2')
% close parallel pool
delete(poolobj)

%% --------------  Quality Check for head motion: view 4D images as animation in mricro GL  -------------- 
for isub = 1:nsub
    fprintf('viewing 4d series for %s\n',ids{isub});
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    check_head_motion(subimg_dir,[-5,4],2.5,true,false,false);
    pause
end

%% --------------  Calculate tSNR for each run of each participants  -------------- 
mask_dir = 'E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\ROIRSA\AALandHCPMMP1';
masks_names = cellstr(spm_select('List',mask_dir,'.*_bilateral.nii'));
masks = cell2struct(fullfile(mask_dir,masks_names),cellfun(@(x) strrep(x,'.nii',''),masks_names,'uni',0));
mean_tsnr = nan(numel(ids)*5,numel(masks_names));
tsnr_dir  = fullfile(qc_dir,'tsnr');
checkdir(tsnr_dir)

%generate image
parfor isub = 1:nsub
    fprintf('calculating tsnr for %s\n',ids{isub});
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    func_imgs   = cellstr(spm_select('FPList',subimg_dir,filepattern.preprocess.normalise));
    for j = 1:numel(func_imgs)
        src_img  = func_imgs{j};
        [~,fn,~] = fileparts(src_img);
        
        outputfn = fullfile(tsnr_dir,['tsnr_',fn,'.nii']);
        calculate_snr(src_img,outputfn);
    end
end

% get the average tsnr within masks
for isub = 1:nsub
    fprintf('calculating tsnr for %s\n',ids{isub});
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    func_imgs   = cellstr(spm_select('FPList',subimg_dir,filepattern.preprocess.normalise));
    for j = 1:numel(func_imgs)
        src_img  = func_imgs{j};
        [~,fn,~] = fileparts(src_img);
        
        outputfn = fullfile(tsnr_dir,['tsnr_',fn,'.nii']);
        Vo = spm_vol(outputfn);
        tmp_tsnr = structfun(@(x) mean(spm_summarise(Vo,x),'all','omitnan'),masks,'uni',0);
        mean_tsnr((isub-1)*5+j,:) = table2array(struct2table(tmp_tsnr));
    end
end
mean_tsnr_T = array2table(mean_tsnr,'VariableNames',masks_names);
tmp = fullfact([5,numel(ids)]);
mean_tsnr_T.subid   = arrayfun(@(k) ids{tmp(k,2)},1:size(tmp,1),'uni',0)';
task_names = {'maintask','maintask','maintask','maintask','localizer'};
task_runs  = [1:4,1];
mean_tsnr_T.taskrun = arrayfun(@(k) task_runs(tmp(k,1)),1:size(tmp,1),'uni',1)';
mean_tsnr_T.taskname = arrayfun(@(k) task_names(tmp(k,1)),1:size(tmp,1),'uni',0)';

mean_tsnr_T.Properties.RowNames = arrayfun(@(k) [ids{tmp(k,2)},'-run',num2str(tmp(k,1))],1:size(tmp,1),'uni',0);
writetable(mean_tsnr_T,fullfile(qc_dir,'QualityCheckLogBook.xlsx'),'Sheet','tsnr_cohort2')