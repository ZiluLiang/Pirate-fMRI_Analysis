% This script runs the univariate PPI analysis on train/test difference
% -----------------------------------------------------------------------    
% Author: Zilu Liang

clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
masks = cellstr(spm_select('FPList','E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\masks','HMAT.*.nii'));
err_tracker   = struct(); %#ok<*UNRCH>
addpath(fullfile(directory.projectdir,'scripts','Exp1_fmri','univariate'))

%% Train-Test GLM with concatenation (because spm ppi can only deal with one session)
flag_runGLM  = true;
if flag_runGLM
    steps = struct('first', {{'specify','concatenate','estimate','contrast'}}, ... #
                   'second',{{'specify','estimate','contrast','result'}});
    glm_name = 'traintest_navigation'; 
    glm_dir = fullfile(directory.fmri_data,strcat(glm_name,'_concated'));
    err_tracker.(glm_name) = glm_runner(glm_name, rmfield(steps,'second'),glm_dir);
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
    %     err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),'','',subid_list,curr_g,subid_Gstr);
    % 
    %     % run second level without covariate FOR G only
    %     err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),'','',gzer_list,sprintf("%sGonly",curr_g));
    % 
    % end
end
%% GLM F-contrast with all regressor of interest
for j = 1:numel(participants.validids)
    subid = participants.validids{j};
    firstlvlsubdir  = fullfile(glm_dir,'first',subid);
    
    Fcontrast_weight = gen_contrast_matrix(fullfile(firstlvlsubdir,'SPM.mat'),struct('training',{1,0},'test',{0,1},'response',{0,0}));
    glm_contrast(firstlvlsubdir,...
                 {'F_traintest_maineff'},...
                 {Fcontrast_weight}, ...
                 false);
    fprintf('%s: Completed first-level F-contrast for %s \n',glm_name,subid)
end

%% Extract VOI
HPCmask = fullfile(directory.fmri_data,"masks/anat_AAL3/HPC_bilateral.nii");
for j = 2:numel(participants.validids)
    subid = participants.validids{j};
    firstlvlsubdir  = fullfile(glm_dir,'first',subid);
    subSPM = load(fullfile(firstlvlsubdir,"SPM.mat")).SPM;
    [contrastidx,~,~] = find_contrast_idx(subSPM,{'F_traintest_maineff'});
    voi_extract(firstlvlsubdir,"HPC_bilateral",contrastidx,HPCmask)

end