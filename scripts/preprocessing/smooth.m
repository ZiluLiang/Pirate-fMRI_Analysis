function smooth(subimg_dir)

    filepattern = get_pirate_defaults(false,'filepattern'); 
    nii_files  = structfun(@(scantype) ...
                                    structfun(@(pattern) ...
                                                  cellstr(spm_select('List', subimg_dir, [pattern,'.*\.nii'])),...
                                              scantype,'UniformOutput',false),...
                               filepattern,'UniformOutput',false);
    smooth = {};
    smooth{1}.spm.spatial.smooth.data(1) = nii_files.preprocess.normalize;
    smooth{1}.spm.spatial.smooth.fwhm = [6 6 6];
    smooth{1}.spm.spatial.smooth.dtype = 0;
    smooth{1}.spm.spatial.smooth.im = 0;
    smooth{1}.spm.spatial.smooth.prefix = 's';
    
    save(fullfile(subimg_dir,'smooth.mat'),'smooth')
    spm_jobman ('run',smooth);
    
end