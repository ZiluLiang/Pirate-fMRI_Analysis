function normalise(subimg_dir)
% normlise functional images
% INPUT:
%  - subimg_dir: directory to participant's preprocessing fmri images
% 
% This will create the following files:
%  w*.nii(smoothed images)
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    % find files
    [fmri,filepattern] = get_pirate_defaults(false,'fmri','filepattern');    
    def_field = cellstr(spm_select('FPList',subimg_dir,filepattern.preprocess.deformation));
    func_imgs = cellstr(spm_select('ExtFPList', subimg_dir, [filepattern.preprocess.realignunwarp,'.*\.nii']));

    % get the bounding box of the template image normalized to
    [bb,~]=spm_get_bbox([spm('dir'),filesep,'tpm',filesep,'TPM.nii']);

    % set up normalization job
    matlabbatch = {};
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = def_field;
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = func_imgs;        
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb  = bb; % change to match the bounding box of the template image
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [fmri.voxelsize fmri.voxelsize fmri.voxelsize]; % change to match the resolution of acquistion
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 7; % change to 7 to achieve higher normalization quality
    
    save(fullfile(subimg_dir,'normalization.mat'),'matlabbatch')
    spm('defaults', 'FMRI');
    spm_jobman('initcfg')
    spm_jobman ('run',matlabbatch);
end