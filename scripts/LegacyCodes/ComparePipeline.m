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
compare_nmi = false; % compare normalization quality
compare_rp = true;% compare head motion estimates

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
    mean_rp_corr = cell(5,1);
    for isub = 1:participants.nsub
        subimg_dir_w  = fullfile(with_reorientation_dir,participants.ids{isub});
        subimg_dir_wo  = fullfile(without_reorientation_dir,participants.ids{isub});
        rp_files_w = cellstr(spm_select('FPList',subimg_dir_w,filepattern.preprocess.motionparam));
        rp_files_wo = cellstr(spm_select('FPList',subimg_dir_wo,filepattern.preprocess.motionparam));
        for j = 1:numel(rp_files_wo)
            rp_res{j}(isub,:) = mean(abs(table2array(subtract_table(readtable(rp_files_w{j}),readtable(rp_files_wo{j})))),1,'omitnan');
            mean_rp_corr{j}(isub,:) = cal_rp_corr(readtable(rp_files_w{j}),readtable(rp_files_wo{j}));
        end
    end
    mean_rp_corr = cellfun(@(x) mean(table2array(x),2,'omitnan'),mean_rp_corr,'uni',0);
    mean_rp_corr{5}(participants.nsub,:) = nan;
    mean_rp_corr = cat(2,mean_rp_corr{:});
    mean_rp_corr = array2table(mean_rp_corr,'VariableNames',{'localizer_run1','navigation_run1','navigation_run2','navigation_run3','navigation_run4'});
    writetable(mean_rp_corr,fullfile(compare_output_dir,'comparepipeline.xlsx'),'Sheet','mu_corr_betweenpipeline')
    
    rp_res = cell2struct(rp_res,{'localizer_run1','navigation_run1','navigation_run2','navigation_run3','navigation_run4'});
    mean_rp_difference = structfun(@(x) array2table(x,'VariableNames',{'x','y','z','pitch','yaw','roll'}),rp_res,'uni',0);
    cellfun(@(f) writetable(mean_rp_difference.(f),fullfile(compare_output_dir,'comparepipeline.xlsx'),'Sheet',['mu_paramdiff-',f]),fieldnames(mean_rp_difference));
    save(fullfile(compare_output_dir,'comparepipeline_RP.mat'),'mean_rp_corr')
    
end

function rho_tab = cal_rp_corr(A,B,varnames)
    if nargin<4, varnames = {'x','y','z','pitch','yaw','roll'}; end
    if nargin<3, rownames = ''; end
    vars = A.Properties.VariableNames;
    rhos = cellfun(@(f) corr(A.(f),B.(f)),vars);
    if isempty(rownames)
        rho_tab  = array2table(rhos, 'VariableNames',varnames);
    else
        rho_tab  = array2table(rhos, 'VariableNames',varnames,'RowNames',rownames);
    end
end

