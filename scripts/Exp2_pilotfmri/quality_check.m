% quality checks of fmri data preprocessing
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO: checkout
% https://github.com/jsheunis/fMRwhy/tree/master/fmrwhy/qcand see if more
% quality control measures should be calculated

%% Configurations
clear;clc
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');                                                         
qc_dir = fullfile(directory.fmri_data,'qualitycheck');
ids           = participants.validids;
nsub          = participants.nvalidsub;

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
%check if signal distortion is better after unwarp
n_tasks = 3;
n_runs  = [1,6,2];
n_volumes = {[174],312*ones(1,6),230*ones(1,2)};
for isub = 1:nsub
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
num_workers   = min([feature('NumCores'),nsub]);
poolobj       =  parpool(num_workers);%set up parallel processing
% run head motion check
hm_dir = fullfile(qc_dir,'headmotion');
hm_tables = cell(nsub,1);
checkdir(hm_dir)
parfor isub = 1:nsub
    fprintf('ploting head motion for %s\n',ids{isub});
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    [hm_tables{isub},f] = check_head_motion(subimg_dir,[-5,4],2.5,false,true,false);
    participant = repmat(ids(isub),size(hm_tables{isub},1),1);
    hm_tables{isub} = addvars(hm_tables{isub},participant,'Before',1);    
    saveas(f,fullfile(hm_dir,[ids{isub},'.png']));
    close all     
end
hm_table = cat(1,hm_tables{:});
writetable(hm_table,fullfile(qc_dir,'QualityCheckLogBook.xlsx'),'Sheet','headmotion')
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
for isub = 1:nsub
    fprintf('calculating tsnr for %s\n',ids{isub});
    subimg_dir  = fullfile(directory.preprocess,ids{isub});
    func_imgs   = cellstr(spm_select('FPList',subimg_dir,filepattern.preprocess.normalise));
    for j = 1:numel(func_imgs)
        src_img  = func_imgs{j};
        [~,fn,~] = fileparts(src_img);
        
        outputfn = fullfile(tsnr_dir,['tsnr_',fn,'.nii']);
        tmp_tsnr   = calculate_snr(src_img,outputfn,masks);
        mean_tsnr((isub-1)*sum(n_runs)+j,:) = table2array(struct2table(tmp_tsnr));
    end
end
mean_tsnr_T = array2table(mean_tsnr,'VariableNames',masks_names);
tmp = fullfact([sum(n_runs),numel(ids)]);
mean_tsnr_T.Properties.RowNames = arrayfun(@(k) [ids{tmp(k,2)},'-run',num2str(tmp(k,1))],1:size(tmp,1),'uni',0);
writetable(mean_tsnr_T,fullfile(qc_dir,'QualityCheckLogBook.xlsx'),'Sheet','tsnr')