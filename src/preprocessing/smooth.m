function smooth(subimg_dir)
% smooth the normlised images using a kernel that equals to twice the voxel size,
% INPUT:
%  - subimg_dir: directory to participant's preprocessing fmri images
% 
% This will create the following files:
%  s*.nii(smoothed images)
% -----------------------------------------------------------------------    
% Author: Zilu Liang


    [fmri,filepattern] = get_pirate_defaults(false,'fmri','filepattern'); 
    func_imgs = cellstr(spm_select('FPList', subimg_dir, [filepattern.preprocess.normalise,'.*\.nii']));
    func_imgs = cellfun(@(fn) cellstr(spm_select('expand',fn)),func_imgs,'uni',0);
    func_imgs = cat(1,func_imgs{:});

    matlabbatch = {};
    matlabbatch{1}.spm.spatial.smooth.data   = func_imgs;
    matlabbatch{1}.spm.spatial.smooth.fwhm   = fmri.voxelsize*2*ones(1,3); %  changed to match 2 times the voxel size
    matlabbatch{1}.spm.spatial.smooth.dtype  = 0;
    matlabbatch{1}.spm.spatial.smooth.im     = 0;
    matlabbatch{1}.spm.spatial.smooth.prefix = 's';
    
    save(fullfile(subimg_dir,'smooth.mat'),'matlabbatch')
    spm('defaults', 'FMRI');
    spm_jobman('initcfg')
    spm_jobman ('run',matlabbatch);
    
end