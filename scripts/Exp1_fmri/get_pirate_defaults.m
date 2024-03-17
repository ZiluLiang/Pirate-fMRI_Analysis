function varargout = get_pirate_defaults(varargin)
% Returns default configurations for pirate fMRI analysis
% FORMAT:
% get_pirate_defaults: return all the configurations in a struct
% get_pirate_defaults(fieldname1,...,fieldnameN): 
%    return specified field of configurations in a struct
% get_pirate_defaults(return_struct_flag,fieldname1,...,fieldnameN):
%    return specified field of configurations
%    if return_struct_flag is true, then a struct is returned,
%    if return_struct_flag is false, then multiple configurations are returned 
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    % set defaults for pirate fMRI analysis
    pirate_defaults = setdefaults();
    
    % set return flags and validate fields
    if numel(varargin) == 0
        return_struct_flag = true;
        get_fields = fieldnames(pirate_defaults);
    else
        if islogical(varargin{1})
            return_struct_flag = varargin{1};
            if numel(varargin) == 1
                get_fields = fieldnames(pirate_defaults);
            else
                get_fields = varargin(2:end);
            end
        else
            return_struct_flag = true;
            get_fields = varargin;
        end
    end
    if ~all(ismember(get_fields,fieldnames(pirate_defaults)))
        error('Fields do not exist! Must be from: %s', strjoin(fieldnames(pirate_defaults),', '))
    end
    
    % parse fields of default settings to output
    cellarr  = cellfun(@(x) pirate_defaults.(x),get_fields,'uni',0);
    if return_struct_flag        
        varargout = {cell2struct(cellarr, get_fields)};
    else
        varargout = cellarr;
    end
end


function pirate_defaults = setdefaults

    pirate_defaults = struct('packages',     struct(),...
                             'directory',    struct(),...
                             'participants', struct(),...
                             'filepattern',  struct(),...
                             'fmri',         struct());

    
    %% --------------  Specify dependencies --------------  
    wk_dir  = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\';
    % packages that do not need to add with subfolders
    pirate_defaults.packages.SPM12        = 'C:\Program Files\MATLAB\matlab toolbox\spm12';
    pirate_defaults.packages.MRIcroGL     = 'C:\MRIcroGL_windows\MRIcroGL\MRIcroGL.exe';
    pirate_defaults.packages.jsonlab      = 'C:\Program Files\MATLAB\matlab toolbox\marsbar-0.45';
    pirate_defaults.packages.marsbar      = 'C:\Program Files\MATLAB\matlab toolbox\jsonlab-master';
    pirate_defaults.packages.SPM12OldNorm = fullfile(pirate_defaults.packages.SPM12,'toolbox','OldNorm'); % this is so that auto reorientation runs
    pirate_defaults.packages.SPM12batch   = fullfile(pirate_defaults.packages.SPM12,'matlabbatch');         % this is so that batch management runs
    % packages that need to add with subfolders
    pirate_defaults.packages.scripts      = genpath(fullfile(wk_dir,'scripts'));
    % add path to packages
    structfun(@(pkg_path) addpath(pkg_path),rmfield(pirate_defaults.packages,'MRIcroGL'))
    
    %% --------------  Specify directory  -------------- 
    pirate_defaults.directory.projectdir   = wk_dir;
    pirate_defaults.directory.pm_default   = fullfile(wk_dir,'scripts','preprocessing','pm_defaults_Prisma_CIMCYC.m');  % specifics for fieldmap 
    pirate_defaults.directory.mni_template = fullfile(spm('dir'),'canonical','avg152T1.nii'); % mni template used for visualization and estimate parameters for auto-reorientation
    pirate_defaults.directory.fmri_data    = fullfile(wk_dir,'data','fmri');
    pirate_defaults.directory.fmribehavior = fullfile(wk_dir,'data','fmri','beh');
    pirate_defaults.directory.preprocess   = fullfile(wk_dir,'data','fmri','preprocess'); % directory to images created during preprocessing
    pirate_defaults.directory.unsmoothed   = fullfile(wk_dir,'data','fmri','unsmoothed'); % directory to preprossesed images without smoothing
    pirate_defaults.directory.smoothed     = fullfile(wk_dir,'data','fmri','smoothed');   % directory to preprossesed images after smoothing
    

    %% --------------  Read subject list -------------- 
    renamer_fn = fullfile(wk_dir,'data','renamer.json');
    renamer    = loadjson(renamer_fn);
    
    pirate_defaults.participants.ids     = fieldnames(renamer);
    pirate_defaults.participants.nsub    = numel(pirate_defaults.participants.ids);
    %%%%%TODO add valid participants list
    pirate_defaults.participants.validids  = pirate_defaults.participants.ids(1:30); % exclude sub 31 due to incomplete scans
    pirate_defaults.participants.nvalidsub = numel(pirate_defaults.participants.validids);% exclude sub 31 due to incomplete scans
    
    pirate_defaults.participants.nonlearnerids     = {'sub010','sub012','sub013','sub027','sub017'}'; 
    pirate_defaults.participants.nongeneralizerids = {'sub010','sub012','sub013','sub027','sub004','sub023','sub002','sub014','sub021'}';
    pirate_defaults.participants.learnerids        = pirate_defaults.participants.validids(~ismember(pirate_defaults.participants.validids,pirate_defaults.participants.nonlearnerids));
    pirate_defaults.participants.generalizerids    = pirate_defaults.participants.validids(~ismember(pirate_defaults.participants.validids,pirate_defaults.participants.nongeneralizerids));

    %% --------------  naming patterns ------------------
    pirate_defaults.filepattern.task1   = 'sub-.*_task-piratenavigation_run-[1-4]';
    pirate_defaults.filepattern.task2   = 'sub-.*_task-localizer_run-[1]';
    % prefix for raw image files (before preprocessing), the fieldnames should not be changed as they will be called in each preprocessing script
    pirate_defaults.filepattern.raw                    = struct('fieldmap',struct(),'anatomical',struct(),'functional',struct());
    %%%%%fieldmaps
    pirate_defaults.filepattern.raw.fieldmap.phasediff = '^sub-.*_fmap-phasediff';
    pirate_defaults.filepattern.raw.fieldmap.shortecho = '^sub-.*_fmap-magnitude1';
    pirate_defaults.filepattern.raw.fieldmap.longecho  = '^sub-.*_fmap-magnitude2';
    %%%%%anatomical scans
    pirate_defaults.filepattern.raw.anatomical.T1      = '^sub-.*_anat-T1w';
    %%%%%functional scans, more fields can be added in the form of task1, task2, task3,... taskn
    pirate_defaults.filepattern.raw.functional.task1   = '^sub-.*_task-piratenavigation_run-[1-4]';% the first run in this will be used as the first session
    pirate_defaults.filepattern.raw.functional.task2   = '^sub-.*_task-localizer_run-[1]';
    %%%%%make sure fields are in the right order
    pirate_defaults.filepattern.raw.functional = orderfields(pirate_defaults.filepattern.raw.functional,...
                                                             arrayfun(@(x) sprintf('task%d',x),1:numel(fieldnames(pirate_defaults.filepattern.raw.functional)),'uni',0));
                                                    
    % prefix for files after reorientation and preprocessing --- these should not be changed as this is set by in each preprocessing script
    pirate_defaults.filepattern.preprocess             = struct('reorient',       '^o',...
                                                                'vdm',            '^vdm5_sc',...
                                                                'firstepiunwarp', '^qc_ufirstepi_',...
                                                                'realignunwarp',  '^u',...
                                                                'meanepi',        '^meanu',... % mean epi created after realign and unwarp
                                                                'meanepi_wu',     '^qc_meanwu',... % mean epi created during quality check on normalized images
                                                                'motionparam',    '^rp_',...
                                                                'coreg',          '^r',...
                                                                'deformation',    '^y_',...
                                                                'normalise',      '^wu',... % normalise adds w, normalise is done on realigned unwarped images, so prefix is wu*
                                                                'normseg_t1',     '^wc',...% normalised and segmented anatomical image
                                                                'smooth',         '^swu',...% smooth adds s, smooth is done on normalized realigned unwarped images, so prefix is swu*
                                                                'nuisance',       '^nuisance_'); % txt files that contains nuisance regressors  
    pirate_defaults.filepattern.reorient               = structfun(@(scantype) ...
                                                                    structfun(@(pattern) [pirate_defaults.filepattern.preprocess.reorient,strrep(pattern,'^','')],scantype,'uni',0),...
                                                                    pirate_defaults.filepattern.raw,'uni',0);
                                                                
    %% --------------  scanning parameters and preprocessing defaults ------------------
    pirate_defaults.fmri.voxelsize = 2.5; % voxel size in mm
    pirate_defaults.fmri.tr        = 1.73;% TR in seconds
    pirate_defaults.fmri.nuisance_terms = {'raw','fw'}; 

    %% --------------  experiment design ------------------
    pirate_defaults.exp.allstim      = 0:24;
    pirate_defaults.exp.trainingstim = [2,7,10,11,12,13,14,17,22];
    pirate_defaults.exp.teststim     = pirate_defaults.exp.allstim(arrayfun(@(x) ~ismember(x,pirate_defaults.exp.trainingstim),pirate_defaults.exp.allstim));

    %% --------------  saving to json file for python reading  ------------------
    %pirate_defaults.participants = rmfield(pirate_defaults.participants,{'ids','nsub'});
    savejson('',pirate_defaults,'FileName',fullfile(wk_dir,'scripts','pirate_defaults.json'),'SingletCell',0,'ForceRootName',0);
end