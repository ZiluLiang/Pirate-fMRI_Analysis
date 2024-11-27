% fmri data preparation: convert, rename to annonymized files and organize
% The script contains four steps controlled by four flags:
%       img_convert, img_organize, img_backup, img_backup
%       img_convert: convert dicom images to nii images
%       img_organize: organize and rename the data according to
%                     renamer.json. organized and renamed files are placed in renamed_dir
%       img_backup: create a backup into preprocess folder for
%                     preprocessing. copied files are placed in preprocess_dir
% 
% Dependencies:
%       - dcm2niix, spm12, jsonlab-master
%       - renamer.json: the renamer.json consists of key-value pairs with new subject 
%                       names as keys, and old subject names as values, old subject names 
%                       are listed in an string array [annonymized id used in behavior file name, fmri file name]
%
%------ written by Zilu Liang (2021.6,BNU)  ------
%------ adapted by Zilu Liang(2023.4,Oxford)------
clear;clc

%% Set up
%----------------- Change the following basic configuration ----------------- 
wk_dir  = 'E:\pirate_fmri\Analysis';
src_dir = fullfile(wk_dir,'src');

raw_data_dir   = 'F:\Exp1_fmri\raw';

study_data_dir   = fullfile(wk_dir,'data','Exp1_fmri');
study_script_dir = fullfile(wk_dir,'scripts','Exp1_fmri');
data_dir       = fullfile(study_data_dir,'fmri');
renamer_fn     = fullfile(study_data_dir,'renamer.json');

dcm2niix_path  = 'C:\MRIcroGL_windows\MRIcroGL\Resources\dcm2niix.exe';
SPM12_dir      = 'C:\Program Files\MATLAB\matlab toolbox\spm12';

%file pattern for different type of scans
filepattern    = struct('fieldmap',         'gre_field_mapping_2.5mm',...
                        't1',               'GIfMI_T1_MPRAGE',...
                        'piratenavigation', 'ep2d_bold_RUN_[1-4]',...
                        'localizer',        'ep2d_bold_RUN_[5]');
% Step control
img_convert  = true;
img_organize = true;
img_backup   = true;

%-------------------------- Do not modify ----------------------------%
converted_dir      = fullfile(data_dir,'converted'); % directory of decompressed files
renamed_dir        = fullfile(data_dir,'renamed');   % directory of renamed files
preprocess_dir     = fullfile(data_dir,'preprocess');% directory of files ready for preprocessing, files are copied from renamed_dir
checkimg_output_fn = 'imagenumercheck.txt';

addpath(genpath(src_dir))
addpath(genpath(study_script_dir))
addpath(SPM12_dir)

%% Read subject list
renamer = loadjson(renamer_fn);
allids = fieldnames(renamer);
% only look for subids in the second batch otherwise set newids = allids
newids = allids(cellfun(@(x) contains(string(x),'sub1'),allids)); 
nsub = numel(newids);

%% Convert images
%converting dicom imgs to nii using dcm2nii command
if img_convert
    for isub = 1:nsub
        new_id = newids{isub};
        old_id = renamer.(new_id){2};
        fprintf('converting sub %s from %s\n',new_id,old_id);
        
        subimg_dir  = fullfile(raw_data_dir,old_id,'Mruz_Chris - 1');
        outimg_dir  = fullfile(converted_dir,new_id);
        checkdir(outimg_dir)
        cd (outimg_dir)
        
        cmd = [dcm2niix_path,' -b y -ba y -g n -o ',['"',outimg_dir,'"'],' ',['"',subimg_dir,'"']];
        %input key-value argument pair used in dcm2niix command
        % -g n: the output files will not be compressed
        % -b y -ba y: generate a BIDS file excluding subject's personal information
        % -o outimg_dir: the converted nii will be saved in outimg_dir
        system(cmd);
    end
end

%% Re-name images and Generate image list
%Rename images and generate a table to check if participant have redundent 
%or missing image files.
%%!!!! check the for redundent images, delete redundant images and re-run
%%!!!! this block before proceeding.
renaming_pattern = struct('fieldmap','fmap-',...
                          't1','anat-T1w',...
                          'piratenavigation','task-piratenavigation_run-[1-4]',...
                          'localizer','task-localizer_run-1');
if img_organize
    fID = fopen(fullfile(data_dir,checkimg_output_fn),'w'); %#ok<*UNRCH>
    fprintf(fID,'%1$s\t%2$s\t%3$s\t%4$s\n','participant','scantype','n_runs','n_trs');
    fclose(fID);   
    
    fileID = fopen(fullfile(data_dir,'imagenumercheck.txt'),'a');
    for isub = 1:nsub
        new_id = newids{isub};
        
        subimg_dir  = fullfile(converted_dir,new_id);
        renamed_subimg_dir  = fullfile(renamed_dir,new_id);
        checkdir(renamed_subimg_dir)
        
        scan_types = fieldnames(filepattern);
        
        for j = 1:numel(scan_types)
            %rename files
            [~,old_names,~] = fileparts(cellstr(spm_select('FPListRec', subimg_dir, ['.*',filepattern.(scan_types{j}),'.*\.nii'])));
            if ~isempty(old_names)
                if ~iscell(old_names)
                    old_names = {old_names};
                end
                new_names = cellfun(@(fn) rename_converted_imgs(scan_types{j},fn,new_id),old_names,'uni',0);
                cd(subimg_dir)
                % rename both json files and nii files
                arrayfun(@(x) copyfile(fullfile(subimg_dir,[old_names{x},'.nii']),fullfile(renamed_subimg_dir,[new_names{x},'.nii'])),1:numel(old_names),'uni',0);
                arrayfun(@(x) copyfile(fullfile(subimg_dir,[old_names{x},'.json']),fullfile(renamed_subimg_dir,[new_names{x},'.json'])),1:numel(old_names),'uni',0);      
            end
            %count files
            new_names = cellstr(spm_select('FPListRec', renamed_subimg_dir, ['.*',renaming_pattern.(scan_types{j}),'.*\.nii']));
            tr_count = strjoin(cellfun(@(fn) num2str(numel(cellstr(spm_select('expand', fn)))),new_names,'uni',0),'+');
            fprintf(fileID,'%1$s\t%2$s\t%3$s\t%4$s\n',new_id,scan_types{j},num2str(numel(new_names)),tr_count);
        end
    end
    fclose(fileID); 
end

%% copy files to a new folder for preprocessing, keep the renamed dir as back up for raw data
if img_backup
    copyfile(renamed_dir,preprocess_dir)
end

%% 
function new_name = rename_converted_imgs(scan_type,old_name,new_id) %#ok<*DEFNU>
    renaming_pattern = struct('fieldmap',         'fmap-%s',...
                              't1',               'anat-T1w',...
                              'piratenavigation', 'task-piratenavigation_run-%s',...
                              'localizer',        'task-localizer_run-1');
    switch scan_type
        case 'fieldmap'
            switch old_name(end-1:end)
                case 'e1'
                    new_name = sprintf(['sub-%s_',renaming_pattern.(scan_type)],new_id(4:end),'magnitude1');
                case 'e2'
                    new_name = sprintf(['sub-%s_',renaming_pattern.(scan_type)],new_id(4:end),'magnitude2');
                case 'ph'
                    new_name = sprintf(['sub-%s_',renaming_pattern.(scan_type)],new_id(4:end),'phasediff');
            end
        case 'piratenavigation'
            parts = strsplit(old_name,'_');
            new_name = sprintf(['sub-%s_',renaming_pattern.(scan_type)],new_id(4:end),parts{8});  %% note that which part correspond to run no depends on the naming rule, double check before proceeding
        otherwise
            new_name = sprintf(['sub-%s_',renaming_pattern.(scan_type)],new_id(4:end));
    end
end