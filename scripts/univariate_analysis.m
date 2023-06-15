clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
%% run Repetition Supression GLMs
RSglm_names = {'rs_loc2d_navigation','rs_resploc2d_navigation','rs_loc2d_localizer','traintest_navigation'};
flag_runGLM  = false;
if flag_runGLM
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:numel(RSglm_names)
        glm_name = RSglm_names{j};
        err_tracker.(glm_name) = run_glm(glm_name, {'spec2est','contrast','second_level'});
    end
end
% extract residuals to double check if models are running okay
for j = 1:numel(RSglm_names)
    glm_name = RSglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name);
end


%% run LSA beta series extrator GLMs - smoothed
LSAglm_names = {'LSA_stimuli_navigation','LSA_stimuli_localizer'};
flag_runGLM  = true;
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
for jdir = 1:numel(lsa_dir)     
    if flag_runGLM
        for j = 1:numel(LSAglm_names)
        glm_name = LSAglm_names{j};
        glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
        checkdir(glm_dir)
        err_tracker.(glm_name) = run_glm(glm_name,{'spec2est'},...
                                         glm_dir,...
                                         directory.smoothed);
        end
    end
end

%% extract residuals to double check if models are running okay
for j = 1:numel(LSAglm_names)
    glm_name = LSAglm_names{j};
    masks = cellstr(spm_select('FPList','D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\wfu','.*.nii'));
    [rangeCon.(glm_name),meanResMS1.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{1},glm_name),masks);
    [rangeCon.(glm_name),meanResMS2.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,lsa_dir{2},glm_name),masks);
end

%% generate contrast for odd and even runs for LSA beta series extractor GLMs
glm_name = 'LSA_stimuli_navigation';
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

%% generate contrast for each stimul for LSA beta series extractor GLMs
glm_name = 'LSA_stimuli_navigation';
lsa_dir = {'unsmoothedLSA','smoothed5mmLSA'};
for jdir = 1:numel(lsa_dir)
    glm_dir = fullfile(directory.fmri_data,lsa_dir{jdir},glm_name);
    for isub  = 1:participants.nvalidsub
        fprintf('Specifying Contrast for %s\n',participants.validids{isub})
        firstlvl_dir = fullfile(glm_dir,'first',participants.validids{isub});
        subSPM = load(fullfile(firstlvl_dir,'SPM.mat'),'SPM').SPM;
        contrast_weights = cell(25,1);
        contrast_names   = cell(25,1);
        for k = 1:25
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