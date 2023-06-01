% written by Zilu Liang (2021.6,BNU)
% adapted by Zillu Liang(2023.4,Oxford)
%%% fmri data preparation: decompress, convert, and organize
clear;clc

%% Set up
%----------------- Change the following basic configuration ----------------- 
script_dir     = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\fmri\scripts';
dcm_dir        = 'D:\OneDrive - Nexus365\MRI_Granada\CHRIS\JIRKO';% directory of dicom files
nii_dir        = 'D:\OneDrive - Nexus365\MRI_Granada\converted\JIRKO';% directory to save nii files converted from dicom

dcm2niix_path  = 'C:\MRIcroGL_windows\MRIcroGL\Resources\dcm2niix.exe';
SPM12_dir      = 'C:\Program Files\MATLAB\matlab toolbox\spm12';
add_path(script_dir,1)
add_path(SPM12_dir,0)

%file pattern for different type of scans                     %'task','ep2d_bold_RUN_[1-8]');%for jirko
filepattern = struct('fieldmap','gre_field_mapping_2.5mm',...
                     't1','GIfMI_T1_MPRAGE',...
                     'task','ep2d_bold_RUN_[1-8]');%for jirko
%                     'navigation','ep2d_bold_RUN_[3-8]',...
%                     'rsa','ep2d_bold_RUN_[1,2,9,10,11]');

% Step control
img_conv   = true;
img_check    = true;

%-------------------------- Do not modify ----------------------------%
checkdir(nii_dir);

%% list all subject 
sublist = {dir(dcm_dir).name};
sublist = sublist(cellfun(@(x) ~contains(x,"."),sublist));
nsub = size(sublist,2);

%% Convert images
%converting dicom imgs to nii using dcm2nii command
if img_conv
   parfor isub = 1:14%1:nsub parfor isub = 1:14
        sub_name = sublist{isub};
        fprintf('converting sub %s \n',sub_name);
        
        subimg_dir  = fullfile(dcm_dir,sub_name,'Mruz_Chris');
        outimg_dir  = fullfile(nii_dir,sub_name);
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

%% Generate image list
%Generate a table to check if participant have redundent or missing image files.
%%!!!! check the for redundent images, delete redundant images and re-run
%%!!!! this block before proceeding.
if img_check
    fID = fopen(fullfile(nii_dir,'imagenumercheck.txt'),'w');
    fprintf(fID,'%1$s\t%2$s\t%3$s\n','participant','scantype','numberofimages');
    fclose(fID);   
    
    fileID = fopen(fullfile(nii_dir,'imagenumercheck.txt'),'a');
    for isub = 1:nsub
        sub_name = sublist{isub};
        subimg_dir  = fullfile(nii_dir,sub_name);
    
        scan_types = fieldnames(filepattern);
        for j = 1:numel(scan_types)
            [~,nii_fns,~] = fileparts(cellstr(spm_select('FPListRec', subimg_dir, ['.*',filepattern.(scan_types{j}),'.*\.nii'])));
            [~,json_fns,~] = fileparts(cellstr(spm_select('FPListRec', subimg_dir, ['.*',filepattern.(scan_types{j}),'.*\.json'])));
            fprintf(fileID,'%1$s\t%2$s\t%3$d\n',sub_name,scan_types{j},numel(cellstr(nii_fns)));
    
        end
    end
    fclose(fileID); 
end