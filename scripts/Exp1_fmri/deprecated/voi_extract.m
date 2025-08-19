function voi_extract(SPMmat_dir,voi_name,contrastidx,maskimg)
    matlabbatch{1}.spm.util.voi.spmmat = cellstr(fullfile(SPMmat_dir,'SPM.mat'));
    matlabbatch{1}.spm.util.voi.adjust = 1;
    matlabbatch{1}.spm.util.voi.session = 1;
    matlabbatch{1}.spm.util.voi.name = char(voi_name);
    matlabbatch{1}.spm.util.voi.roi{1}.spm.spmmat = {''};
    matlabbatch{1}.spm.util.voi.roi{1}.spm.contrast = contrastidx;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.threshdesc = 'FWE';
    matlabbatch{1}.spm.util.voi.roi{1}.spm.thresh = 0.05;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.extent = 0;
    matlabbatch{1}.spm.util.voi.roi{2}.mask.image = cellstr(maskimg);
    matlabbatch{1}.spm.util.voi.roi{2}.mask.threshold = 0.5;
    matlabbatch{1}.spm.util.voi.expression = 'i1 & i2';
    
    spm_jobman('run',matlabbatch);

end