% This script runs the LSA glms for extracting beta series, and concatenate
% the beta series for further MVPA (RSA&decoding) analysis in python
% -----------------------------------------------------------------------    
% Author: Zilu Liang

clear;clc

[directory,participants,filepattern,exp]  = get_pirate_defaults(false,'directory','participants','filepattern','exp');

addpath(genpath(fullfile(directory.projectdir,"scripts","Exp1_fmri")))
subidlist       = participants.validids;%intersect(participants.cohort2ids,participants.validids);%participants.validids;
nsub            = numel(subidlist);

%% run LSA beta series extrator GLMs
LSAglm_names = {'LSA_stimuli_navigation','LSA_stimuli_localizer'};
LSAglm_names = {'LSA_stimuli_navigation_movangle'};
flag_runGLM  = true;
lsa_dir      = 'unsmoothedLSA';
preproc_dir  = directory.unsmoothed;
steps        = struct('first', {{'specify','estimate','contrast'}}, ...
                      'second',{{'specify','estimate','contrast','result'}});
steps        = struct('first', {{'contrast'}}, ...
                      'second',{{'specify','estimate','contrast','result'}});
if flag_runGLM
    for j = 1:numel(LSAglm_names) %#ok<*UNRCH>
        glm_name = LSAglm_names{j};
        glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);
        checkdir(glm_dir)

        err_tracker.(glm_name) = glm_runner(glm_name,...
                                            rmfield(steps,'second'),...
                                            glm_dir,...
                                            preproc_dir, ...
                                            subidlist, ...
                                            '','','', ...
                                            true);
    end
end

%% run second level

% glm_name = LSAglm_names{j};
% glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);
% 
% pgroups = struct('C1C2', {participants.validids},...
%                  'C1',{participants.cohort1ids}, ...
%                  'C2',{participants.cohort2ids});
% gnames = fieldnames(pgroups);
% SGnames = {'nG','G'};
% for kg = 1:numel(gnames)
%     curr_g = gnames{kg};
%     subid_list = participants.validids(cellfun(@(x) ismember(x,pgroups.(curr_g)), participants.validids));
%     subid_Gstr = cellfun(@(x) SGnames{1+ismember(x,participants.generalizerids)}, subid_list,'UniformOutput',false);
%     gzer_list  = participants.generalizerids(cellfun(@(x) ismember(x, subid_list), participants.generalizerids));
%     ngzer_list = participants.nongeneralizerids(cellfun(@(x) ismember(x, subid_list), participants.nongeneralizerids));
% 
%     % run second level without covariate FOR ALL
%     err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),glm_dir,'',subid_list,curr_g,subid_Gstr);
% 
%     % run second level without covariate FOR G only
%     err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),glm_dir,'',gzer_list,sprintf("%sGonly",curr_g));
% 
% end

%% extract residuals to double check if models are running okay
glm_name = 'LSA_stimuli_navigation_movdir1col2pmod';
masks = cellstr(spm_select('FPList','E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\ROIRSA\AALandHCPMMP1bilateral','.*_bilateral.nii'));
check_contrasts= {glm_configure(glm_name).contrasts.name};

[rangeCon1.(glm_name),meanCon1.(glm_name),meanResMS1.(glm_name),rangeStat1.(glm_name),meanStat1.(glm_name)] = ...
    extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir,glm_name), ...
                             repmat(masks([1,3]),numel({check_contrasts}),1),check_contrasts,...
                             participants.validids);
glm_name0 = 'LSA_stimuli_navigation';
check_contrasts0= {glm_configure(glm_name).contrasts.name};
[rangeCon1.(glm_name0),meanCon1.(glm_name0),meanResMS1.(glm_name0),rangeStat1.(glm_name0),meanStat1.(glm_name0)] = ...
    extract_firstlvl_spmStat(glm_name0,fullfile(directory.fmri_data,lsa_dir,glm_name0), ...
                             repmat(masks([1,3]),numel({check_contrasts0}),1),check_contrasts0,...
                             participants.validids);

mrs1 = meanResMS1.(glm_name).HPC_bilateral;
tsnr_imgs = [cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-1.nii",sid(4:6)),subidlist,'UniformOutput',true)),...
             cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-2.nii",sid(4:6)),subidlist,'UniformOutput',true)),...
             cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-3.nii",sid(4:6)),subidlist,'UniformOutput',true)),...
             cellstr(cellfun(@(sid) sprintf("tsnr_wuosub-%s_task-piratenavigation_run-4.nii",sid(4:6)),subidlist,'UniformOutput',true))];
tsnr_imgs = fullfile(directory.fmri_data,"qualitycheck","tsnr",tsnr_imgs);
tsnr_means = arrayfun(@(j) mean(spm_summarise(cellstr(tsnr_imgs(j,:)),masks{1},'mean')), 1:1:nsub)';

figure
scatter(tsnr_means,mrs1)
ylabel('mean residual'); xlabel('mean tsnr')


task_imgs = cellfun(@(sid) cellstr(spm_select('FPList',fullfile(directory.unsmoothed,sid),'wuosub-.*_task-piratenavigation_run.*')),subidlist,'uni',0);
bold_means = arrayfun(@(j) mean(spm_summarise(task_imgs{j},masks{2},'mean')), 1:1:nsub)';
figure
scatter(bold_means,mrs1)
ylabel('mean residual'); xlabel('mean raw bold signal')

figure
histogram(mrs1./bold_means)


%% concatenate regressor/contrast estimates into 4D series - navigation task
% reg img - 100 regressors one for each stimuli in each run
%    concatenated into-> stimuli_4r.nii (no averaging)

glm_name = 'LSA_stimuli_navigation_movangle';
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


% %% concatenate into 4D series - localizer task
% % 9 regressors one for each training stimuli
% glm_name = 'LSA_stimuli_localizer';
% lsa_dir = 'unsmoothedLSA';
% 
% glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);
% for isub  = 1:nsub
%     fprintf('Concatenating 4D activity pattern images for %s\n',subidlist{isub})
%     firstlvl_dir = fullfile(glm_dir,'first',subidlist{isub});
%     subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
%     [~,reg_img] = arrayfun(@(stimid) find_regressor_idx(subSPM,sprintf('stim%02d',stimid)),exp.trainingstim);
% 
%     fprintf('%s - reg_img: %d/9\n', subidlist{isub},numel(reg_img))
%     if all(cellfun(@(x) ~isempty(x),reg_img),"all")
%         % ordered by stimid: stim[2,7,10,11,12,13,14,17,22]
%         spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_1r.nii'));
%         fprintf('Completed concatenating 4D activity pattern images for %s\n',subidlist{isub})
%     else
%         error('failed to find enough file to concatenate')            
%     end
%     clear firstlvl_dir subSPM reg_img
% end


%% concatenate regressor/contrast estimates into 4D series - navigation task with and without response
% reg img - 100 regressors one for each stimuli in each run
%    concatenated into-> stimuli_4r.nii (no averaging)

glm_name = 'LSA_stimuli_navigation_respsplit';
lsa_dir = 'unsmoothedLSA';
glm_dir = fullfile(directory.fmri_data,lsa_dir,glm_name);

for isub  = 1:nsub
    fprintf('Concatenating 4D activity pattern images for %s\n',subidlist{isub})
    firstlvl_dir = fullfile(glm_dir,'first',subidlist{isub});
    subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
    reg_img_withwo_resp = cell(2,numel(exp.allstim),4);
    for k = 1:numel(exp.allstim)
        % find the index of regressor
        [~,reg_img_withwo_resp(1,k,:)] = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02dresp',runid,exp.allstim(k))),1:numel(subSPM.Sess));
        [~,reg_img_withwo_resp(2,k,:)] = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02dnoresp',runid,exp.allstim(k))),1:numel(subSPM.Sess));
    end
    
    
    if all(cellfun(@(x) ~isempty(x),reg_img_withwo_resp),"all")
        % ordered by run then stimid:  stim00r1 -- stim24r1 -- stim00r2 -- stim24r2 ...          
        tmp = permute(squeeze(reg_img_withwo_resp(1,:,:)),[1,2]);
        reg_img = vertcat(tmp(:));
        spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_4r_resp.nii'));
        fprintf('Completed concatenating 4D activity pattern images for response trials %s\n',subidlist{isub})

        tmp = permute(squeeze(reg_img_withwo_resp(2,:,:)),[1,2]);
        reg_img = vertcat(tmp(:));
        spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_4r_noresp.nii'));
        fprintf('Completed concatenating 4D activity pattern images for no response trials %s\n',subidlist{isub})
    else            
        error('failed to find enough file to concatenate:\n %s - reg_img: %d/100\n',...
                subidlist{isub},...
                numel(reg_img_withwo_resp(cellfun(@(x) ~isempty(x),reg_img_withwo_resp))))
    end
    clear firstlvl_dir subSPM contrast_img1 contrast_imgo contrast_imge reg_img tmp
end


%% clear paths
rmpath(genpath(fullfile(directory.projectdir,"scripts","Exp1_fmri")))