% This script runs the univariate analysis on repetition suppression and
% neural axis and train/test difference
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO need to rerun to test if new code is bug-free
clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
masks = cellstr(spm_select('FPList','E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\masks','HMAT.*.nii'));
err_tracker   = struct(); %#ok<*UNRCH>
addpath(fullfile(directory.projectdir,'scripts','Exp1_fmri','univariate'))

%% Check number of repetition suppression event in each participant
%checkvars = {'onset_training','onset_test'};
%checkvarnames = checkvars;
checkvars = {'onset_rstrials','onset_rstraining','onset_rstest','onset_rstraintest'};
checkvarnames = {"repetition suppression", "train-train RS", "test-test RS", "train-test RS"};
trial_count = zeros([numel(participants.validids),4,numel(checkvars)]);
for isub = 1:numel(participants.validids)
    sid = participants.validids{isub};
    for run = 1:4
        data_filename = sprintf('sub-%s_task-piratenavigation_run-%d.mat',...
                            strrep(sid,'sub',''), run);
        rundata = load(fullfile(directory.fmribehavior,sid,data_filename)).data;
        %
        trial_count(isub,run,:) = cellfun(@(cvar) sum(~isnan(rundata.(cvar))),checkvars);

    end
end

figure 
tiles =  tiledlayout(numel(checkvars),4);
tiles.TileSpacing = 'loose';
for c=1:numel(checkvars)
    for run = 1:4
        currax = nexttile(tiles);
        histogram(trial_count(:,run,c));
        title(sprintf('%s-run%d',checkvarnames{c},run));
        xlabel("N trial"); ylabel("N sub")
    end
end
% to check if the last three sum up to the first one
trial_count_seprs = sum(trial_count,[3]) - trial_count(:,:,1);
all((trial_count_seprs - trial_count(:,:,1))==0)
%% run Repetition Supression and train-test GLMs
RSglm_names = {
    'traintest_navigation','traintest_navigation_wvsworesp',...
    'rs_loc2d_localizer','rs_loc2d_navigation','rs_loc2dsepgroup_navigation'};%,... 
    %'rs_feacture2d_navigation','rs_color_navigation','rs_shape_navigation',...
    %'rs_hrchydist_navigation','rs_hrchydistucord_navigation','rs_hrchydistquadr_navigation'...
    %};
flag_runGLM  = true;
if flag_runGLM
    steps = struct('first', {{'specify','estimate','contrast'}}, ... #
                   'second',{{'specify','estimate','contrast','result'}});
    for j = 1:numel(RSglm_names)
        glm_name = RSglm_names{j};        
        err_tracker.(glm_name) = glm_runner(glm_name, rmfield(steps,'second'));
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
            err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),'','',subid_list,curr_g,subid_Gstr);

            % run second level without covariate FOR G only
            err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),'','',gzer_list,sprintf("%sGonly",curr_g));
            
        end
    end
end
%% extract residuals to double check if models are running okay
for j = 1:numel(RSglm_names)
    glm_name = RSglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,'',masks);
end

%% run neural-axis analysis
NAglm_names = {'axis_loc_navigation','axis_locsepgroup_navigation','axis_loc_localizer','axis_loc_wprevtrial_localizer',...
    'dist2train_navigation','dist2train_navigation_traintest','axis_resploc_navigation_traintest'};
flag_runGLM  = true;
if flag_runGLM
    steps = struct('first', {{'specify','estimate','contrast'}}, ...
                   'second',{{'specify','estimate','contrast'}});
    for j = 6:7%4:numel(NAglm_names)
        glm_name = NAglm_names{j};
        err_tracker.(glm_name) = glm_runner(glm_name, rmfield(steps,'second'));
        cd(fullfile(directory.projectdir,'scripts'))
        if exist(fullfile(directory.fmri_data,glm_name,'second'),"dir")
            rmdir(fullfile(directory.fmri_data,glm_name,'second'),'s')
        end
        glm_runner(glm_name, ...
                   rmfield(steps,'first'), ...
                   '','','', ...
                   'allparticipants');
        glm_runner(glm_name, ...
                   rmfield(steps,'first'), ...
                   '','', ...
                   [participants.generalizerids;participants.nongeneralizerids], ...
                   'generalizer_vs_nongeneralizer', ...
                   cellstr([repmat("G",size(participants.generalizerids));repmat("NG",size(participants.nongeneralizerids))]))
    end
end
%% extract residuals to double check if models are running okay
for j = 1:numel(NAglm_names)
    glm_name = NAglm_names{j};
    [rangeCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,'',masks);
end