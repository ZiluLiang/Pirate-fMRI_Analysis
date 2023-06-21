clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
participants.nonlearnerids     = {'sub010','sub012','sub013','sub027','sub017'}; 
participants.nongeneralizerids = {'sub010','sub012','sub013','sub027','sub004','sub023','sub002','sub014','sub021','sub017'};
participants.learnerids        = participants.validids(~ismember(participants.validids,participants.nonlearnerids));
participants.generalizerids    = participants.validids(~ismember(participants.validids,participants.nongeneralizerids));


%% run Repetition Supression GLMs
RSglm_names = {'rs_loc2d_navigation','rs_resploc2d_navigation','rs_loc2d_localizer','traintest_navigation'};
flag_runGLM  = true;
if flag_runGLM
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:numel(RSglm_names)
        glm_name = RSglm_names{j};
        err_tracker.(glm_name) = run_glm(glm_name, {'spec2est','contrast','second_level'});
        run_glm(glm_name,{'second_level'},'','',participants.learnerids,'learner_');
        run_glm(glm_name,{'second_level'},'','',participants.generalizerids,'generalizer_');
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(RSglm_names)
    glm_name = RSglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
end


%% run LSA beta series extrator GLMs
LSAglm_names = {'LSA_stimuli_navigation','LSA_stimuli_localizer','LSA_stimuli_navigation_modeltraintest'};
flag_runGLM  = true;
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
preproc_dir = {directory.unsmoothed,directory.smoothed};
for jdir = 1:numel(lsa_dir)     
    if flag_runGLM
        for j = 3%1:numel(LSAglm_names)
            glm_name = LSAglm_names{j};
            glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
            checkdir(glm_dir)
            err_tracker.(glm_name) = run_glm(glm_name,{'spec2est'},...
                                             glm_dir,...
                                             preproc_dir{jdir}, ...
                                             participants.validids);
        end
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(LSAglm_names)
    glm_name = LSAglm_names{j};
    masks = cellstr(spm_select('FPList','D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\wfu','.*.nii'));
    [rangeCon.(glm_name),meanResMS1.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{1},glm_name),masks);
    [rangeCon.(glm_name),meanResMS2.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{2},glm_name),masks);
end


%% run neural-axis analysis
NAglm_names = {'axis_loc_navigation','axis_resploc_navigation','axis_loc_localizer','axis_attrloc_navigation','axis_attryloc_navigation'};
flag_runGLM  = true;
if flag_runGLM
    for j = numel(NAglm_names)
        glm_name = NAglm_names{j};
        err_tracker.(glm_name) = run_glm(glm_name, {'spec2est','contrast','second_level'});
        run_glm(glm_name,{'second_level'});
        run_glm(glm_name,{'second_level'},'','',participants.learnerids,'learner_');
        run_glm(glm_name,{'second_level'},'','',participants.generalizerids,'generalizer_');
    end
end
%extract residuals to double check if models are running okay
for j = 1:numel(NAglm_names)
    glm_name = NAglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
end

%% generate contrast for odd and even runs for LSA beta series extractor GLMs - navigation task
glm_name = 'LSA_stimuli_navigation';%'LSA_stimuli_navigation';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
allstimid = 0:24;
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Specifying Contrast for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        contrast_weights = cell(25,2);
        contrast_names   = cell(25,2);
        for k = 1:25
            % initialize a weight vector for contrast
            init_wvec = zeros(size(subSPM.xX.name));
            % find the index of regressor
            runstim_idx = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02d',runid,allstimid(k))),1:numel(subSPM.Sess));
            % set odd runs stim reg to 1
            con_odd = init_wvec;
            con_odd(runstim_idx(1:2:end)) = 1;
            % set even runs stim reg to 1
            con_even = init_wvec;
            con_even(runstim_idx(2:2:end)) = 1;
            contrast_weights{k,1} = con_odd;
            contrast_weights{k,2} = con_even;
            contrast_names(k,:) = {sprintf('stim%02d_odd',allstimid(k)),sprintf('stim%02d_even',allstimid(k))};
        end
        
        contrast_weights = reshape(contrast_weights,numel(contrast_weights),1);
        contrast_names = reshape(contrast_names,numel(contrast_names),1);
        specify_estimate_contrast(firstlvl_dir,...
                                  contrast_names,...
                                  contrast_weights);
        clear firstlvl_dir subSPM contrast_names contrast_weights
        fprintf('Completed specifying Contrast for %s\n',participants.validids{isub})
    end
end

%% generate contrast for each stimul for LSA beta series extractor GLMs - navigation task
glm_name = 'LSA_stimuli_navigation';%'LSA_stimuli_navigation';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
allstimid = 0:24;
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Specifying Contrast for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        contrast_weights = cell(25,1);
        contrast_names   = cell(25,1);
        for k = 1:numel(allstimid)
            % initialize a weight vector for contrast
            con_stim = zeros(size(subSPM.xX.name));
            % find the index of regressor
            runstim_idx = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02d',runid,allstimid(k))),1:numel(subSPM.Sess));
            % set stim reg to 1
            con_stim(runstim_idx) = 1;
            contrast_weights{k,1} = con_stim;
            contrast_names(k,:) = {sprintf('stim%02d',allstimid(k))};
        end    
        contrast_weights = reshape(contrast_weights,numel(contrast_weights),1);
        contrast_names = reshape(contrast_names,numel(contrast_names),1);
        specify_estimate_contrast(firstlvl_dir,...
                                  contrast_names,...
                                  contrast_weights, ...
                                  false);% do not replace existing contrast
        clear firstlvl_dir subSPM contrast_names contrast_weights
        fprintf('Completed specifying Contrast for %s\n',participants.validids{isub})
    end
end

%% concatenate into 4D series - navigation task
% 1 - 25 contrasts one for each stimuli
% 2 - 50 contrasts one for each stimuli in odd/even run
% 3 - 100 regressors one for each stimuli in each run
glm_name = 'LSA_stimuli_navigation';%'LSA_stimuli_navigation';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
allstimid = 0:24;
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        [contrast_img1,contrast_imgo,contrast_imge] = deal(cell(numel(allstimid),1));
        reg_img = cell(numel(allstimid),4);
        for k = 1:numel(allstimid)
            % find contrast image 1
            [~,contrast_img1{k},~] = find_contrast_idx(subSPM,regexpPattern(sprintf('^stim%02d$',allstimid(k))));
            
            % find contrast image 2
            [~,contrast_imgo{k},~] = find_contrast_idx(subSPM,sprintf('stim%02d_odd',allstimid(k)));
            [~,contrast_imge{k},~] = find_contrast_idx(subSPM,sprintf('stim%02d_even',allstimid(k)));
            
            % find the index of regressor
            [~,reg_img(k,:)] = arrayfun(@(runid) find_regressor_idx(subSPM,sprintf('Sn(%d) stim%02d',runid,allstimid(k))),1:numel(subSPM.Sess));
        end
        
        fprintf('%s - contrast_img1: %d/25, contrast_img2: %d/50, reg_img: %d/100\n',...
            participants.validids{isub},...
            numel(contrast_img1),...
            numel([contrast_imgo;contrast_imge]),...
            numel(reg_img))
        if ~any(cellfun(@isempty,[contrast_img1;contrast_imgo;contrast_imge;reshape(reg_img,[],1)]))
            % ordered by stim00 -- stim24
            spm_file_merge(char(fullfile(firstlvl_dir,contrast_img1)),fullfile(firstlvl_dir,'stimuli_mu.nii'));
            
            % ordered by stim00odd -- stim24odd -- stim00even -- stim24even
            spm_file_merge(char(fullfile(firstlvl_dir,[contrast_imgo;contrast_imge])),fullfile(firstlvl_dir,'stimuli_oe.nii'));

            % ordered by stim00r1 -- stim24r1 -- stim00r2 -- stim24r2 ...        
            tmp = permute(reg_img,[1,2]);
            reg_img = vertcat(tmp(:));
            spm_file_merge(char(fullfile(firstlvl_dir,reg_img)),fullfile(firstlvl_dir,'stimuli_4r.nii'));
            fprintf('Completed concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        else
            error('failed to find enough file to concatenate')            
        end
        clear firstlvl_dir subSPM contrast_img1 contrast_imgo contrast_imge reg_img tmp
    end
end

%% concatenate into 4D series - localizer task
% 9 regressors one for each training stimuli
glm_name = 'LSA_stimuli_localizer';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
trainingstimid = [2,7,10,11,12,13,14,17,22];
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Concatenating 4D activity pattern images for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        [~,reg_img] = arrayfun(@(stimid) find_regressor_idx(subSPM,sprintf('stim%02d',stimid)),trainingstimid);
        
        fprintf('%s - reg_img: %d/9\n', numel(reg_img))
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