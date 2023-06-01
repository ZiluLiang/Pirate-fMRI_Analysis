% written by Zilu Liang (2021.6,BNU)
% adapted by Zilu Liang (2023.3)
%%% fmri data preprocessing

clear;clc

%% Set up
%----------------- Change the following basic configuration ----------------- 
wk_dir         = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\fmri';
script_dir     = fullfile(wk_dir,'scripts');
SPM12_dir      = 'C:\Program Files\MATLAB\matlab toolbox\spm12';
add_path(script_dir,1)
add_path(SPM12_dir,0)

data_dir = fullfile(wk_dir,'data','fmri');

% Step control
setpath = false;
img_conv   = false;
img_arr = false;

%do not modify
converted_dir = fullfile(data_dir,'converted');
preproc_dir  = fullfile(data_dir,'preprocessing');

%% run preprocess
task_json_file = fullfile(data_dir,'pipeline_preproc_test.json');
tasks = loadjson(task_json_file);
tasks.source = converted_dir;
spm_task = struct();
spm_task.naming = tasks.naming;
spm_task.subjects = tasks.subjects;
spm_task.source = tasks.source;
spm_task.start = false;
spm_task.pattern = '^sub.*_bold.*\.nii$';
spm_task.spm = tasks.modules.fieldmap;
fieldmap(spm_task)

spm_task.start = false;
spm_task.spm = struct();
spm_task.pattern = '^sub.*_bold.*\.nii$';
realign_unwarp(spm_task); 