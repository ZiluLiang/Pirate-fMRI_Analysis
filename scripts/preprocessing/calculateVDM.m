function calculateVDM(subimg_dir,varargin)
% calculate voxel distortion map
%
% output:
%  vdm5_sc*:
%  wfmag_*: forward warpped magnitude image
% ------ written by Zillu Liang(2023.4,Oxford)------
    
    % get flags
    use_reorient = true;
    if numel(varargin) == 1 && islogical(use_reorient)
        use_reorient = varargin{1};
    end

    % get regular expression for different image files 
    [filepattern,directory] = get_pirate_defaults(false,'filepattern','directory');
    
    % get reoriented images, or raw images if re-oriented images are not found or explicitly sepecified that reoriented images are not to be used
    if use_reorient
        nii_files   = structfun(@(scantype) ...
                                  structfun(@(pattern) spm_select('FPList', subimg_dir, [pattern,'.*\.nii']),...
                                             scantype,'UniformOutput',false),...
                               filepattern.reorient,...
                               'UniformOutput',false);
    end
    if use_reorient && all(structfun(@(scantype) all(structfun(@(scans) ~isempty(scans),scantype)),nii_files))
        nii_files = structfun(@(scantype) structfun(@(scans) cellstr(scans),scantype,'uni',0),nii_files,'uni',0);
    else
        nii_files   = structfun(@(scantype) ...
                                  structfun(@(pattern) cellstr(spm_select('FPList', subimg_dir, [pattern,'.*\.nii'])),...
                                             scantype,'UniformOutput',false),...
                               filepattern.raw,...
                               'UniformOutput',false);
    end
    
    % set up vdm
    first_epi = strrep(nii_files.functional.task1{1},'.nii','.nii,1');% use the first volume of the functional images here just for quality inspection, please see SPM instruction
    matlabbatch = {};
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.data.presubphasemag.phase = nii_files.fieldmap.phasediff;
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.data.presubphasemag.magnitude = nii_files.fieldmap.shortecho; %select the one with short echo
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsfile  = cellstr(directory.pm_default);
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.session.epi = cellstr(first_epi); 
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.matchvdm = 1; % cogregister the magnitude image to the EPI image.
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.sessname = 'session';
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.writeunwarped = 1; % save the unwarped image for quality check
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.anat = nii_files.anatomical.T1;
    matlabbatch{1}.spm.tools.fieldmap.calculatevdm.subj.matchanat = 1;
    
    save(fullfile(subimg_dir,'calculate_VDM.mat'),'matlabbatch')
    spm('defaults', 'FMRI');
    spm_jobman('initcfg')
    spm_jobman ('run',matlabbatch);
    
    % rename the unwarped first epi file so that it doesn't interfere with
    % file selction in the later realign and unwarp step
    [~, firstepi_name, ~] = fileparts(nii_files.functional.task1{1});
    old_unwarped_epi_name = spm_select('List',subimg_dir,['^u',firstepi_name]);
    new_unwarped_epi_name = ['qc_ufirstepi_',firstepi_name,'.nii'];
    movefile(fullfile(subimg_dir,old_unwarped_epi_name),...
             fullfile(subimg_dir,new_unwarped_epi_name))
    
end


    