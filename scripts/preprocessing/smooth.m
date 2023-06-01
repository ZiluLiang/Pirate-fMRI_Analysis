function smooth(subimg_dir)

    [fmri,filepattern] = get_pirate_defaults(false,'fmri','filepattern'); 
    func_imgs = cellstr(spm_select('FPList', subimg_dir, [filepattern.preprocess.normalise,'.*\.nii']));
    func_imgs = cellfun(@(fn) cellstr(spm_select('expand',fn)),func_imgs,'uni',0);
    func_imgs = cat(1,func_imgs{:});

    smooth = {};
    smooth{1}.spm.spatial.smooth.data   = func_imgs;
    smooth{1}.spm.spatial.smooth.fwhm   = fmri.voxelsize*2*ones(1,3); %  changed to match 2 times the voxel size
    smooth{1}.spm.spatial.smooth.dtype  = 0;
    smooth{1}.spm.spatial.smooth.im     = 0;
    smooth{1}.spm.spatial.smooth.prefix = 's';
    
    save(fullfile(subimg_dir,'smooth.mat'),'smooth')
    spm_jobman ('run',smooth);
    
end