% This script runs the LSA glms for extracting beta series, and concatenate
% the beta series for further MVPA (RSA&decoding) analysis in python
% -----------------------------------------------------------------------    
% Author: Zilu Liang

clear;clc

[directory,participants,filepattern,exp]  = get_pirate_defaults(false,'directory','participants','filepattern','exp');
masks = cellstr(spm_select('FPList','E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\ROIRSA\AALandHCPMMP1bilateral','.*_bilateral.nii'));

addpath(genpath(fullfile(directory.projectdir,"scripts","Exp1_fmri")))
subidlist       = participants.validids;
nsub            = numel(subidlist);

%% run LSA beta series extrator GLMs
LSAglm_names = {'LSA_stimuli_navigation','LSA_stimuli_localizer'};
flag_runGLM  = true;
lsa_dir      = 'unsmoothedLSA';
preproc_dir  = directory.unsmoothed;
steps        = struct('first', {{'specify','estimate','contrast'}}, ...
                      'second',{{'specify','estimate','contrast','result'}});
if flag_runGLM
    for j = 1:numel(LSAglm_names) %#ok<*UNRCH>
        glm_name = LSAglm_names{j};
        glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);
        checkdir(glm_dir)

        err_tracker.(glm_name) = glm_runner(glm_name,...
                                            rmfields(steps,'second'),...
                                            glm_dir,...
                                            preproc_dir, ...
                                            subidlist, ...
                                            '','','', ...
                                            true);
    end
end

%% run second level

glm_name = 'LSA_stimuli_navigation';
glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);

pgroups = struct('C1C2', {participants.validids},...
                 'C1',{participants.cohort1ids}, ...
                 'C2',{participants.cohort2ids});
gnames = fieldnames(pgroups);
SGnames = {'nG','G'};
for kg = 1:numel(gnames)
    curr_g = gnames{kg};
    subid_list = participants.validids(cellfun(@(x) ismember(x,pgroups.(curr_g)), participants.validids));
    subid_Gstr = cellfun(@(x) SGnames{1+ismember(x,participants.generalizerids)}, subid_list,'UniformOutput',false);
    gzer_list  = participants.generalizerids(cellfun(@(x) ismember(x, subid_list), participants.generalizerids));
    ngzer_list = participants.nongeneralizerids(cellfun(@(x) ismember(x, subid_list), participants.nongeneralizerids));
    
    % run second level without covariate FOR ALL
    err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),glm_dir,'',subid_list,curr_g,subid_Gstr);

    % run second level without covariate FOR G only
    err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),glm_dir,'',gzer_list,sprintf("%sGonly",curr_g));
    
end

%% extract residuals to double check if models are running okay
glm_name = 'LSA_stimuli_navigation';
check_contrasts= {'stimuli','training','test','train_min_test','test_min_train'};
[rangeCon1.(glm_name),meanCon1.(glm_name),meanResMS1.(glm_name),rangeStat1.(glm_name),meanStat1.(glm_name)] = ...
    extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir,glm_name), ...
                             repmat(masks(1:2),numel({check_contrasts}),1),check_contrasts,...
                             participants.validids);

% [~,~,meanResMS2.(glm_name),rangeStat2.(glm_name),meanStat2.(glm_name)] = ...
%     extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir,sprintf('%sC1',glm_name)), ...
%                              repmat(masks(1:2),numel({check_contrasts}),1),check_contrasts, ...
%                              participants.cohort1ids(cellfun(@(x) ismember(x,participants.validids),participants.cohort1ids)));

mrs1 = meanResMS1.LSA_stimuli_navigation.HPC_bilateral;
tsnr_imgs = [cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-1.nii",sid(4:6)),subidlist,'UniformOutput',true)),...
             cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-2.nii",sid(4:6)),subidlist,'UniformOutput',true)),...
             cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-3.nii",sid(4:6)),subidlist,'UniformOutput',true)),...
             cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-4.nii",sid(4:6)),subidlist,'UniformOutput',true))];
tsnr_imgs = fullfile(directory.fmri_data,"qualitycheck","tsnr",tsnr_imgs);
tsnr_means = arrayfun(@(j) mean(spm_summarise(cellstr(tsnr_imgs(j,:)),masks{1},'mean')), 1:1:nsub)';

figure
scatter(tsnr_means,mrs1)
ylabel('mean residual'); xlabel('mean tsnr')

figure
task_imgs = cellfun(@(sid) cellstr(spm_select('FPList',fullfile(directory.unsmoothed,sid),'wuosub-.*_task-piratenavigation_run.*')),subidlist,'uni',0);
bold_means = arrayfun(@(j) mean(spm_summarise(task_imgs{j},masks{1},'mean')), 1:1:nsub)';
scatter(bold_means,mrs1)
ylabel('mean residual'); xlabel('mean raw bold signal')

figure
histogram(mrs1./bold_means)

% figure
% mrs2 = meanResMS2.LSA_stimuli_navigation.HPC_bilateral;
% scatter(mrs1(1:numel(mrs2)),mrs2)
% xlabel("mrs1");ylabel("mrs2")
% hold on
% line(mrs1,mrs1)
% hold on
% line(mrs1,0.5*mrs1)
% 
% figure
% conname = 'stimuli'
% con1 = meanStat1.LSA_stimuli_navigation.(conname).HPC_bilateral;
% con2 = meanStat2.LSA_stimuli_navigation.(conname).HPC_bilateral;
% scatter(con1(1:numel(con2)),con2)
% xlabel("con1");ylabel("con2")
% hold on
% line(con1,con1)
%% concatenate regressor/contrast estimates into 4D series - navigation task
% contrast img group 1 - 25 contrasts one for each stimuli
%    concatenated into-> stimuli_all.nii (each stimulus's average effect across all runs)
% contrast img group 2 - two 25 contrasts one for each stimuli in odd/even run 
%    concatenated into-> stimuli_odd.nii and stimuli_even.nii (each stimulus's average effect across odd/even runs)
% reg img - 100 regressors one for each stimuli in each run
%    concatenated into-> stimuli_4r.nii (no averaging)

glm_name = 'LSA_stimuli_navigation';
lsa_dir = 'unsmoothedLSA';
glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);

for isub  = 1:nsub
    fprintf('Concatenating 4D activity pattern images for %s\n',subidlist{isub})
    firstlvl_dir = fullfile(glm_dir,'first',subidlist{isub});
    subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
    reg_img = cell(numel(exp.allstim),4);
    for k = 1:numel(exp.allstim)
        % find the index of regressor
        [~,reg_img(k,:)] = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02d',runid,exp.allstim(k))),1:numel(subSPM.Sess));
    end
    
    
    if all(cellfun(@(x) ~isempty(x),reg_img),"all")
        % ordered by run then stimid:  stim00r1 -- stim24r1 -- stim00r2 -- stim24r2 ...        
        tmp = permute(reg_img,[1,2]);
        reg_img = vertcat(tmp(:));
        spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_4r.nii'));
        fprintf('Completed concatenating 4D activity pattern images for %s\n',subidlist{isub})
    else            
        error('failed to find enough file to concatenate:\n %s - reg_img: %d/100\n',...
                subidlist{isub},...
                numel(reg_img(cellfun(@(x) ~isempty(x),reg_img))))
    end
    clear firstlvl_dir subSPM contrast_img1 contrast_imgo contrast_imge reg_img tmp
end


%% concatenate into 4D series - unconcatenated glms localizer task
% 9 regressors one for each training stimuli
glm_name = 'LSA_stimuli_localizer';
lsa_dir = 'unsmoothedLSA';

glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);
for isub  = 1:nsub
    fprintf('Concatenating 4D activity pattern images for %s\n',subidlist{isub})
    firstlvl_dir = fullfile(glm_dir,'first',subidlist{isub});
    subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
    [~,reg_img] = arrayfun(@(stimid) find_regressor_idx(subSPM,sprintf('stim%02d',stimid)),exp.trainingstim);
    
    fprintf('%s - reg_img: %d/9\n', subidlist{isub},numel(reg_img))
    if all(cellfun(@(x) ~isempty(x),reg_img),"all")
        % ordered by stimid: stim[2,7,10,11,12,13,14,17,22]
        spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_1r.nii'));
        fprintf('Completed concatenating 4D activity pattern images for %s\n',subidlist{isub})
    else
        error('failed to find enough file to concatenate')            
    end
    clear firstlvl_dir subSPM reg_img
end


%% clear paths
rmpath(genpath(fullfile(directory.projectdir,"scripts","Exp1_fmri")))