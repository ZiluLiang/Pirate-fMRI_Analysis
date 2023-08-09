% This script runs the LSA glms for extracting beta series, and concatenate
% the beta series for further MVPA (RSA&decoding) analysis in python
% -----------------------------------------------------------------------    
% Author: Zilu Liang

clear;clc

[directory,participants,filepattern,exp]  = get_pirate_defaults(false,'directory','participants','filepattern','exp');
masks = cellstr(spm_select('FPList','D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat','.*_bilateral.nii'));

%% run LSA beta series extrator GLMs (unconcatenated)
LSAglm_names = {'LSA_stimuli_navigation','LSA_stimuli_localizer'};
flag_runGLM  = true;
lsa_dir      = {'unsmoothedLSA','smoothed5mmLSA'};
preproc_dir  = {directory.unsmoothed,directory.smoothed};
steps        = struct('first', {{'specify','estimate','contrast'}});
for jdir = 1:numel(lsa_dir)     
    if flag_runGLM
        for j = 1:numel(LSAglm_names) %#ok<*UNRCH>
            glm_name = LSAglm_names{j};
            glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
            checkdir(glm_dir)
            err_tracker.(glm_name) = glm_runner(glm_name,...
                                                steps,...
                                                glm_dir,...
                                                preproc_dir{jdir}, ...
                                                participants.validids);
        end
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(LSAglm_names)
    glm_name = LSAglm_names{j};
    [~,~,meanResMS1.(glm_name),rangeStat1.(glm_name),meanStat1.(glm_name)] = ...
        extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{1},glm_name), ...
                                 repmat(masks(1),numel({glm_configure(glm_name).contrasts.name}),1));

    [~,~,meanResMS2.(glm_name),rangeStat2.(glm_name),meanStat2.(glm_name)] = ...
        extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{2},glm_name), ...
                                 repmat(masks(1),numel({glm_configure(glm_name).contrasts.name}),1));
end

%% concatenate regressor/contrast estimates into 4D series - unconcatenated glms navigation task
% contrast img group 1 - 25 contrasts one for each stimuli
%    concatenated into-> stimuli_all.nii (each stimulus's average effect across all runs)
% contrast img group 2 - two 25 contrasts one for each stimuli in odd/even run 
%    concatenated into-> stimuli_odd.nii and stimuli_even.nii (each stimulus's average effect across odd/even runs)
% reg img - 100 regressors one for each stimuli in each run
%    concatenated into-> stimuli_4r.nii (no averaging)

glm_name = 'LSA_stimuli_navigation';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        [contrast_img1,contrast_imgo,contrast_imge] = deal(cell(numel(exp.allstim),1));
        reg_img = cell(numel(exp.allstim),4);
        for k = 1:numel(exp.allstim)
            % find contrast image group 1
            [~,contrast_img1{k},~] = find_contrast_idx(subSPM,regexpPattern(sprintf('^stim%02d$',exp.allstim(k))));
            
            % find contrast image group 2
            [~,contrast_imgo{k},~] = find_contrast_idx(subSPM,sprintf('stim%02d_odd',exp.allstim(k)));
            [~,contrast_imge{k},~] = find_contrast_idx(subSPM,sprintf('stim%02d_even',exp.allstim(k)));
            
            % find the index of regressor
            [~,reg_img(k,:)] = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02d',runid,exp.allstim(k))),1:numel(subSPM.Sess));
        end
        
        
        if ~any(cellfun(@isempty,[contrast_img1;contrast_imgo;contrast_imge;reshape(reg_img,[],1)]))
            % ordered by stim00 -- stim24
            spm_file_merge(char(fullfile(firstlvl_dir,contrast_img1)),fullfile(firstlvl_dir,'stimuli_all.nii'));
            
            % ordered by stim00odd -- stim24odd/stim00even -- stim24even
            spm_file_merge(char(fullfile(firstlvl_dir,contrast_imgo)),fullfile(firstlvl_dir,'stimuli_odd.nii'));
            spm_file_merge(char(fullfile(firstlvl_dir,contrast_imge)),fullfile(firstlvl_dir,'stimuli_even.nii'));

            % ordered by stim00r1 -- stim24r1 -- stim00r2 -- stim24r2 ...        
            tmp = permute(reg_img,[1,2]);
            reg_img = vertcat(tmp(:));
            spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_4r.nii'));
            fprintf('Completed concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        else            
            error('failed to find enough file to concatenate:\n %s - contrast_img1: %d/25, contrast_img2: %d/50, reg_img: %d/100\n',...
                    participants.validids{isub},...
                    numel(contrast_img1),...
                    numel([contrast_imgo;contrast_imge]),...
                    numel(reg_img))
        end
        clear firstlvl_dir subSPM contrast_img1 contrast_imgo contrast_imge reg_img tmp
    end
end

%% concatenate into 4D series - unconcatenated glms localizer task
% 9 regressors one for each training stimuli
glm_name = 'LSA_stimuli_localizer';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        [~,reg_img] = arrayfun(@(stimid) find_regressor_idx(subSPM,sprintf('stim%02d',stimid)),exp.trainingstim);
        
        fprintf('%s - reg_img: %d/9\n', participants.validids{isub},numel(reg_img))
        if all(~isempty(reg_img))
            % ordered by stim[2,7,10,11,12,13,14,17,22]
            spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_1r.nii'));
            fprintf('Completed concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        else
            error('failed to find enough file to concatenate')            
        end
        clear firstlvl_dir subSPM reg_img
    end
end

%% run LSA beta series extrator GLMs for navigation task (concatentaed)
LSAglm_names = {'LSA_stimuli_navigation_concatall','LSA_stimuli_navigation_concatodd','LSA_stimuli_navigation_concateven'};
flag_runGLM  = true;
lsa_dir      = {'unsmoothedLSA','smoothed5mmLSA'};
preproc_dir  = {directory.unsmoothed,directory.smoothed};
steps        = struct('first', {{'specify','concatenate','estimate','contrast'}});
for jdir = 1:numel(lsa_dir)     
    if flag_runGLM
        for j = 1:numel(LSAglm_names)
            glm_name = LSAglm_names{j};
            glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
            checkdir(glm_dir)
            err_tracker.(glm_name) = glm_runner(glm_name,...
                                                steps,...
                                                glm_dir,...
                                                preproc_dir{jdir}, ...
                                                participants.validids);
        end
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(LSAglm_names)
    glm_name = LSAglm_names{j};
    [~,~,meanResMS1.(glm_name),rangeStat1.(glm_name),meanStat1.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{1},glm_name),masks);
    [~,~,meanResMS2.(glm_name),rangeStat2.(glm_name),meanStat2.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{2},glm_name),masks);
end

%% concatenate regressor estimates into 4D series - concatenated glms navigation task
% contrast img group 1 - 25 regressors one for each stimuli in the concatall glm
%    concatenated into-> stimuli_all.nii (each stimulus's average effect across all runs)
% contrast img group 2 - two 25 regressors one for each stimuli in concatodd/concateven glm
%    concatenated into-> stimuli_odd.nii and stimuli_even.nii (each stimulus's average effect across odd/even runs)

lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
for jdir = 1:numel(lsa_dir)
    concat_glm_dir = struct('all',  fullfile(directory.fmri_data,lsa_dir{jdir},'LSA_stimuli_navigation_concatall'), ...
                            'odd',  fullfile(directory.fmri_data,lsa_dir{jdir},'LSA_stimuli_navigation_concatodd'), ...
                            'even', fullfile(directory.fmri_data,lsa_dir{jdir},'LSA_stimuli_navigation_concateven'));
    for isub  = 1:participants.nvalidsub
        fprintf('Concatenating 4D activity pattern images for %s\n',participants.validids{isub})

        firstlvl_dir = structfun(@(x) fullfile(x,'first',participants.validids{isub}),concat_glm_dir,'UniformOutput',false);
        subSPM       = structfun(@(x) load(fullfile(x,'SPM.mat'),'SPM').SPM,firstlvl_dir,'UniformOutput',false);        

        [~,reg_img] = structfun(@(sSPM) ...
                    arrayfun(@(k) find_regressor_idx(subSPM.all,sprintf('stim%02d',k)),exp.allstim),...
                    subSPM,"UniformOutput",false);

        fnames = fieldnames(concat_glm_dir);
        if ~any(structfun(@(x) any(cellfun(@isempty,x)),reg_img))
            % ordered by stim00 -- stim24
            cellfun(@(x) ...
                spm_file_merge(char(fullfile(firstlvl_dir.(x),reg_img.(x))),...
                               fullfile(firstlvl_dir.(x),sprintf('stimuli_%s.nii',x))...
                              ),...
                fnames);
            fprintf('Completed concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        else
            failed_folders = cellfun(@(x) concat_glm_dir.(x),fnames(structfun(@(x) any(cellfun(@isempty,x)),reg_img)),'uni',0);
            error('failed to find enough file to concatenate in the following folders: \n%s\n', ...
                strjoin(failed_folders,'\n')...
                )            
        end
        clear firstlvl_dir subSPM reg_img
    end
end
