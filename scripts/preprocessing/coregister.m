function coregister(subimg_dir,use_reorient)
% coregister the T1(source) to the mean epi image(reference)
% INPUT:
%  - subimg_dir: directory to participant's preprocessing fmri images
%  - use_reorient: use reoriented or raw image for preprocessing
% 
% This will create the following files:
%  r*(a coregistered T1 image)
% -----------------------------------------------------------------------    
% Author: Zilu Liang


    % get flags
    if nargin<2 || ~islogical(use_reorient), use_reorient = true; end

    % get regular expression for different image files 
    filepattern = get_pirate_defaults(false,'filepattern');
    
    % get reoriented images, or raw images if re-oriented images are not found or explicitly sepecified that reoriented images are not to be used
    if use_reorient
        nii_files = structfun(@(scantype) ...
                                  structfun(@(pattern) spm_select('FPList', subimg_dir, [pattern,'.*\.nii']),...
                                             scantype,'UniformOutput',false),...
                               filepattern.reorient,...
                               'UniformOutput',false);
    end
    if use_reorient && all(structfun(@(scantype) all(structfun(@(scans) ~isempty(scans),scantype)),nii_files))
        nii_files = structfun(@(scantype) structfun(@(scans) cellstr(scans),scantype,'uni',0),nii_files,'uni',0);
    else
        nii_files = structfun(@(scantype) ...
                                  structfun(@(pattern) cellstr(spm_select('FPList', subimg_dir, [pattern,'.*\.nii'])),...
                                             scantype,'UniformOutput',false),...
                               filepattern.raw,...
                               'UniformOutput',false);
    end
    meanepi_img = cellstr(spm_select('FPList', subimg_dir, [filepattern.preprocess.meanepi,'.*\.nii']));
    
    matlabbatch  = {};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = nii_files.anatomical.T1;
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = meanepi_img; 
    matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
%     Defaults from spm
%     coregister{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
%     coregister{1}.spm.spatial.coreg.estimate.eoptions.sep      = [4 2];
%     coregister{1}.spm.spatial.coreg.estimate.eoptions.tol      = [ 0.0200 0.0200 0.0200 0.0010 0.0010 0.0010 ...
%                                                                    0.0100 0.0100 0.0100 0.0010 0.0010 0.0010];
%     coregister{1}.spm.spatial.coreg.estimate.eoptions.fwhm     = [7 7];        
    
    save(fullfile(subimg_dir,'coregister.mat'),'matlabbatch')
    spm('defaults', 'FMRI');
    spm_jobman('initcfg')
    spm_jobman ('run',matlabbatch);
end