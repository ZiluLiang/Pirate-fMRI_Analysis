% compare the normalization between pipelines with and without reorientation
% this is done by comparing the normalized mutual information measure
% computed as in spm_coreg (whitout smoothing the histograms)
clear;clc
%% Configurations
[directory,participants,filepattern] = get_pirate_defaults(false,'directory','participants','filepattern');                                                         

with_reorientation_dir = directory.preprocess;
without_reorientation_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\preprocess-automated';

compare_output_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\qualitycheck\compare_pipeline';
checkdir(compare_output_dir)

%flags
compare_nmi = true; % compare normalization quality
compare_rp = false;% compare head motion estimates

%% compare normalization quality
if compare_nmi
    % loop over : 1- w / wo reorientation, 2- task1 / task2, 3- sub
    regexp_epis = {[filepattern.preprocess.normalise,'.*',strrep(filepattern.raw.functional.task1,'^','')],...
                   [filepattern.preprocess.normalise,'.*',strrep(filepattern.raw.functional.task2,'^','')]};
    regexp_mepi = {'meanepi_task1',...
                   'meanepi_task2'};           
    n_task = numel(regexp_epis);
    nmi_res = cell(n_task,1); %#ok<*UNRCH>
    for t = 1:n_task
        nmi_res{t} = nan(participants.nsub,2);
        for isub = 1:participants.nsub
            subimg_dir_w  = fullfile(with_reorientation_dir,participants.ids{isub});
            subimg_dir_wo  = fullfile(without_reorientation_dir,participants.ids{isub});

            if isempty(spm_select('FPList',compare_output_dir,['w_',regexp_mepi{t},'_',participants.ids{isub}]))
                meanepi_task_w = create_mean_epi(subimg_dir_w,regexp_epis{t},['w_',regexp_mepi{t},'_',participants.ids{isub}],compare_output_dir);
            else
                meanepi_task_w = cellstr(spm_select('FPList',compare_output_dir,['w_',regexp_mepi{t},'_',participants.ids{isub}]));
            end

            if isempty(spm_select('FPList',compare_output_dir,['wo_',regexp_mepi{t},'_',participants.ids{isub}]))
                meanepi_task_wo = create_mean_epi(subimg_dir_wo,regexp_epis{t},['wo_',regexp_mepi{t},'_',participants.ids{isub}],compare_output_dir);
            else
                meanepi_task_wo = cellstr(spm_select('FPList',compare_output_dir,['wo_',regexp_mepi{t},'_',participants.ids{isub}]));
            end

            %template_img = 'C:\MRIcroGL_windows\MRIcroGL\mni_icbm152_nl_VI_nifti\icbm_avg_152_t1_tal_nlin_symmetric_VI.nii';
            template_img = fullfile(spm('Dir'),'canonical','avg152T1.nii');
            source_img = [meanepi_task_w,meanepi_task_wo];
            nmi_res{t}(isub,:) = ext_spm_coregcost(source_img,template_img);
        end
    end
    nmi_res_T = array2table(cat(2,nmi_res{:}),'VariableNames',{'navigation_reoriented','navigation_noreorient','localizer_reoriented','localizer_noreorient'});
    writetable(nmi_res_T,fullfile(compare_output_dir,'comparepipeline.xlsx'),'Sheet','NMI_of_meanepi')
end

%% compare head motion estimates
if compare_rp
    regexp_rps = {[filepattern.preprocess.motionparam,'.*',strrep(filepattern.raw.functional.task1,'^','')],...
                   [filepattern.preprocess.motionparam,'.*',strrep(filepattern.raw.functional.task2,'^','')]};
    n_task = numel(regexp_rps);
    rp_res = cell(5,1); %#ok<*UNRCH>
    mean_xyz_distance = cell(5,1);
    for isub = 1:participants.nsub
        subimg_dir_w  = fullfile(with_reorientation_dir,participants.ids{isub});
        subimg_dir_wo  = fullfile(without_reorientation_dir,participants.ids{isub});
        rp_files_w = cellstr(spm_select('FPList',subimg_dir_w,filepattern.preprocess.motionparam));
        rp_files_wo = cellstr(spm_select('FPList',subimg_dir_wo,filepattern.preprocess.motionparam));
        for j = 1:numel(rp_files_wo)
            rp_res{j}(isub,:) = mean(abs(table2array(subtract_table(readtable(rp_files_w{j}),readtable(rp_files_wo{j})))),1,'omitnan');
            mean_xyz_distance{j}(isub,:) = cal_rp_xyz_dist(readtable(rp_files_w{j}),readtable(rp_files_wo{j}));
        end
    end
    mean_xyz_distance = cellfun(@(x) mean(x,2,'omitnan'),mean_xyz_distance,'uni',0);
    mean_xyz_distance{5}(participants.nsub,:) = nan;
    mean_xyz_distance = cat(2,mean_xyz_distance{:});
    mean_xyz_distance = array2table(mean_xyz_distance,'VariableNames',{'localizer_run1','navigation_run1','navigation_run2','navigation_run3','navigation_run4'});
    rp_res = cell2struct(rp_res,{'localizer_run1','navigation_run1','navigation_run2','navigation_run3','navigation_run4'});
    mean_rp_difference = structfun(@(x) array2table(x,'VariableNames',{'x','y','z','pitch','yaw','roll'}),rp_res,'uni',0);
    save(fullfile(compare_output_dir,'comparepipeline_RP.mat'),'mean_xyz_distance','mean_rp_difference')
    writetable(mean_xyz_distance,fullfile(compare_output_dir,'comparepipeline.xlsx'),'Sheet','mu_XYZdistance')
    cellfun(@(f) writetable(mean_rp_difference.(f),fullfile(compare_output_dir,'comparepipeline.xlsx'),'Sheet',['mu_paramdiff-',f]),fieldnames(mean_rp_difference));
end

function distance = cal_rp_xyz_dist(A,B)
    matA = table2array(A);
    matB = table2array(B);
    distance = arrayfun(@(k) norm(matA(k,1:3) - matB(k,1:3)),1:size(A,1));    
end

