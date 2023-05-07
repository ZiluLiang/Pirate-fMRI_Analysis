function normalise(subimg_dir)
    % find files
    filepattern = get_pirate_defaults(false,'filepattern');    
    def_field = cellstr(spm_select('FPList',subimg_dir,filepattern.preprocess.deformation));
    func_imgs = cellstr(spm_select('ExtFPList', subimg_dir, [filepattern.preprocess.realignunwarp,'.*\.nii']));

    % get the bounding box of the template image normalized to
    [bb,~]=spm_get_bbox([spm('dir'),filesep,'tpm',filesep,'TPM.nii']);

    % set up normalization job
    normalization = {};
    normalization{1}.spm.spatial.normalise.write.subj.def = def_field;
    normalization{1}.spm.spatial.normalise.write.subj.resample = func_imgs;        
    normalization{1}.spm.spatial.normalise.write.woptions.bb  = bb; % change to match the bb of the template image
    normalization{1}.spm.spatial.normalise.write.woptions.vox = [2.5 2.5 2.5]; % change to match the resolution of acquistion
    normalization{1}.spm.spatial.normalise.write.woptions.interp = 4; % change to 7 to achieve 
    
    save(fullfile(subimg_dir,'normalization.mat'),'normalization')
    spm_jobman ('run',normalization);
end