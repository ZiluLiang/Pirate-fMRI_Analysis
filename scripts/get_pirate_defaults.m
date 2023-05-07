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
% written by Zillu Liang(2023.5,Oxford)

    % set defaults for pirate fMRI analysis
    pirate_defaults = setdefaults();
    
    % set return flags and validate fields
    if numel(varargin) == 0
        return_struct_flag = true;
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
        varargout = cell2struct(cellarr, get_fields);
    else
        varargout = cellarr;
    end
end


function pirate_defaults = setdefaults

    pirate_defaults = struct('directory',struct(),...
                             'participants',struct(),...
                             'filepattern',struct(),...
                             'handles',struct());

    
    %% --------------  Specify directory  --------------  
    wk_dir          = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\';
    script_dir      = fullfile(wk_dir,'scripts');
    SPM12_dir       = 'C:\Program Files\MATLAB\matlab toolbox\spm12';
    add_path(script_dir,1)
    add_path(SPM12_dir,0)

    pirate_defaults.directory.MRIcroGL     = 'C:\MRIcroGL_windows\MRIcroGL\MRIcroGL.exe';
    pirate_defaults.directory.pm_default   = fullfile(script_dir,'preprocessing','pm_defaults_Prisma_CIMCYC.m');  % specifics for fieldmap 
    pirate_defaults.directory.mni_template = 'C:\MRIcroGL_windows\MRIcroGL\mni_icbm152_nl_VI_nifti\icbm_avg_152_t1_tal_nlin_symmetric_VI.nii'; % mni template used for visualization and estimate parameters for auto-reorientation
    pirate_defaults.directory.preprocess   = fullfile(wk_dir,'data','fmri_image','preprocess'); % directory to images created during preprocessing
    pirate_defaults.directory.unsmoothed   = fullfile(wk_dir,'data','fmri_image','unsmoothed'); % directory to preprossesed images without smoothing
    pirate_defaults.directory.smoothed     = fullfile(wk_dir,'data','fmri_image','smoothed'); % directory to preprossesed images after smoothing
    

    %% --------------  Read subject list -------------- 
    renamer_fn = fullfile(wk_dir,'data','renamer.json');
    renamer    = loadjson(renamer_fn);
    
    pirate_defaults.participants.ids     = fieldnames(renamer);
    pirate_defaults.participants.nsub    = numel(pirate_defaults.participants.ids);
    %%%%%TODO add valid participants list
    
    
    %% --------------  naming patterns ------------------
    % prefix for raw image files (before preprocessing), the fieldnames should not be changed as they will be called in each preprocessing script
    pirate_defaults.filepattern.raw                    = struct('fieldmap',struct(),'anatomical',struct(),'functional',struct());
    %fieldmaps
    pirate_defaults.filepattern.raw.fieldmap.phasediff = '^sub-.*_fmap-phasediff';
    pirate_defaults.filepattern.raw.fieldmap.shortecho = '^sub-.*_fmap-magnitude1';
    pirate_defaults.filepattern.raw.fieldmap.longecho  = '^sub-.*_fmap-magnitude2';
    %anatomical scans
    pirate_defaults.filepattern.raw.anatomical.T1      = '^sub-.*_anat-T1w';
    %functional scans, more fields can be added in the form of task1, task2, task3,... taskn
    pirate_defaults.filepattern.raw.functional.task1   = '^sub-.*_task-piratenavigation_run-[1-4]';% the first run in this will be used as the first session
    pirate_defaults.filepattern.raw.functional.task2   = '^sub-.*_task-localizer_run-[1]';
    %make sure fields are in the right order
    pirate_defaults.filepattern.raw.functional = orderfields(pirate_defaults.filepattern.raw.functional,...
                                                             arrayfun(@(x) sprintf('task%d',x),1:numel(fieldnames(pirate_defaults.filepattern.raw.functional)),'uni',0));
                                                    
    % prefix for files after reorientation and preprocessing --- these should not be changed as this is set by in each preprocessing script
    pirate_defaults.filepattern.reorient               = structfun(@(scantype) ...
                                                                    structfun(@(pattern) ['^o',strrep(pattern,'^','')],scantype,'uni',0),...
                                                                    pirate_defaults.filepattern.raw,'uni',0);
    pirate_defaults.filepattern.preprocess             = struct('vdm',            '^vdm5_sc',...
                                                                'firstepiunwarp', '^qc_ufirstepi_',...
                                                                'realignunwarp',  '^u',...
                                                                'meanepi',        '^meanu',... % mean epi created after realign and unwarp
                                                                'meanepi_wu',     '^qc_meanwu',... % mean epi created during quality check on normalized images
                                                                'motionparam',    '^rp_',...
                                                                'coreg',          '^r',...
                                                                'deformation',    '^y_',...
                                                                'normalise',      '^wu',... % normalise adds w, normalise is done on realigned unwarped images, so prefix is wu*
                                                                'normseg_t1',     '^wc',...% normalised and segmented anatomical image
                                                                'smooth',         '^swu'); % smooth adds s, smooth is done on normalized realigned unwarped images, so prefix is swu*
    %%%%%TODO add behavior as well

    
    %% --------------  funtion handles ------------------
    % functions called to perform preprocessing step
    pirate_defaults.handles.preprocess  = struct('reorient',       @reorient,...
                                                 'calVDM',         @calculateVDM,...
                                                 'realign_unwarp', @realign_unwarp,...
                                                 'coregistration', @coregister,...
                                                 'segmentation',   @segment,...
                                                 'normalisation',  @normalise,...
                                                 'smooth',         @smooth);


end

function add_path(varargin)
% add_path(path1,path2,...,pathn,add_with_subfolders_flag)
% path1, ..., pathn: paths to add
% add_with_subfolders_flag : 1-add path with subfolders, 0-add without subfolders
    if nargin == 1
        add_with_subfolders_flag = 0;
        new_paths = varargin(1);
    else
        if isnumeric(varargin{end})
            new_paths = varargin(1:end-1);
            add_with_subfolders_flag = varargin{end};            
        end
    end

    for j = 1:numel(new_paths)
        new_path = new_paths{j};
        if ~contains([pathsep, path, pathsep], [pathsep, new_path, pathsep], 'IgnoreCase', ispc)
            if add_with_subfolders_flag
                addpath(genpath(new_path));
            else
                addpath(new_path);
            end
        end
    end
end