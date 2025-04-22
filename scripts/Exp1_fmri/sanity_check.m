% sanity check GLMs of fmri data preprocessing
% after glm estimation:
% check visual and motor effect at group level
% check visual and motor effect at first level by extracting statistics
% from visual and motor ROI
% -----------------------------------------------------------------------    
% Author: Zilu Liang


clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
glm_names = {'sc_navigation','sc_localizer'};

flag_runGLM = true;
flag_getStat = true;

addpath(genpath(fullfile(directory.projectdir,"scripts","Exp1_fmri")))
%% run sanity check GLMs
if flag_runGLM
    err_tracker   = struct(); %#ok<*UNRCH>
    for j = 1:numel(glm_names)
        glm_name = glm_names{j};
        steps  = struct('first', {{'contrast'}}, ... %'specify','estimate',
                        'second',{{'specify','estimate','contrast','result'}});
        glm_dir  = fullfile(directory.fmri_data,glm_name);
        err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'second'));

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
            % err_tracker.(glm_name) = glm_runner(glm_name,rmfield(steps,'first'),'','',gzer_list,sprintf("%sGonly",curr_g));
        end
    end
end

%% examine sanity check results - extract stats
masks = {fullfile('E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\masks','HCPV1V2_bilateral.nii'),...
         fullfile('E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\masks','HMAT_Motor.nii')};
if flag_getStat
    rangeCon = struct();
    meanResMS = struct();
    rangeStat = struct();
    for j = 1%:numel(glm_names)
        glm_name = glm_names{j};
        [rangeCon.(glm_name),meanCon.(glm_name),meanResMS.(glm_name),rangeStat.(glm_name),meanStat.(glm_name)] = extract_firstlvl_spmStat(glm_name,fullfile(directory.fmri_data,glm_name),masks);
        rangeStat.(glm_name).Properties.RowNames = participants.validids;
    end
end