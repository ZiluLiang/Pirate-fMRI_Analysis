% compare the normalization between pipelines with and without reorientation
% this is done by comparing the normalized mutual information measure
% computed as in spm_coreg (whitout smoothing the histograms)
clear;clc
%% Configurations
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');                                                         

with_reorientation_dir = directory.preprocess;
without_reorientation_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothed';

compare_output_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\qualitycheck\compare_pipeline';
checkdir(compare_output_dir)

%flags
compare_nmi = true; % compare normalization quality
compare_rp = false;% compare head motion estimates

%% compare normalization quality
if compare_nmi
    % loop over : 1- w / wo reorientation, 2- task1 / task2, 3- sub
    regexp_epis = {[filepattern.preprocess.normalise,'o.*',strrep(filepattern.raw.functional.task1,'^','')],...
                   [filepattern.preprocess.normalise,'o.*',strrep(filepattern.raw.functional.task2,'^','')]};
    regexp_mepi = regexp_epis;           
    n_task = numel(regexp_epis);
    nmi_res = cell(n_task,1); %#ok<*UNRCH>
    sum_res = nan(numel(1:10:290),2);
    for k = 1:10:290
    for t = 1:n_task
        nmi_res{t} = nan(participants.nsub,2);
        for isub = 1:participants.nsub
            subimg_dir_w  = fullfile(with_reorientation_dir,participants.ids{isub});
            subimg_dir_wo  = fullfile(without_reorientation_dir,participants.ids{isub});

            meanepi_task_w = cellstr(spm_select('FPList',subimg_dir_w,regexp_mepi{t}));
            
            meanepi_task_wo = cellstr(spm_select('FPList',subimg_dir_wo,regexp_mepi{t}));
           
            %template_img = 'C:\MRIcroGL_windows\MRIcroGL\mni_icbm152_nl_VI_nifti\icbm_avg_152_t1_tal_nlin_symmetric_VI.nii';
            template_img = fullfile(spm('Dir'),'canonical','avg152T1.nii');
            source_img = [meanepi_task_w(1),meanepi_task_wo(1)];
            nmi_res{t}(isub,:) = ext_spm_coregcost(source_img,template_img,'nmi',k);
        end
    end
    nmi_res_T = array2table(cat(2,nmi_res{:}),'VariableNames',{'navigation_7bs','navigation_4bs','localizer_7bs','localizer_4bs'});
    fprintf('navigation: %.4f\n',sum(nmi_res_T.navigation_7bs -nmi_res_T.navigation_4bs))
    fprintf('localizer: %.4f\n',sum(nmi_res_T.localizer_7bs -nmi_res_T.localizer_4bs))
    sum_res(k,:) = [mean(nmi_res_T.navigation_7bs -nmi_res_T.navigation_4bs),...
                    mean(nmi_res_T.localizer_7bs -nmi_res_T.localizer_4bs)];
    %pause
   end
end
