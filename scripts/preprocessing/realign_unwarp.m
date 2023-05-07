function realign_unwarp(subimg_dir,varargin)
% realign the first scan of each session to the first scan of the first session, 
% realign the other scans to the first scan within each session,
% and unwarp using the vdm calculated from fieldmap images
%
% output: u*(realigned and unwarped epi images), mean*(mean epi image)
% ------ written by Zilu Liang(2023.4,Oxford)------
    
    % get flags
    use_reorient = true;
    if numel(varargin) == 1 && islogical(use_reorient)
        use_reorient = varargin{1};
    end

    % get regular expression for different image files 
    filepattern = get_pirate_defaults(false,'filepattern');
    
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
    vdm_img = cellstr(spm_select('FPList', subimg_dir, [filepattern.preprocess.vdm,'.*\.nii']));
    func_imgs = struct2cell(nii_files.functional);
    func_imgs = cat(1,func_imgs{:});
    
    % set realign_unwarp job
    realign_unwarp = {};
    for sess = 1:numel(func_imgs)
        realign_unwarp{1}.spm.spatial.realignunwarp.data(sess).scans = cellstr(spm_select('expand',func_imgs{sess}));
        realign_unwarp{1}.spm.spatial.realignunwarp.data(sess).pmscan = vdm_img;
    end
    
    save(fullfile(subimg_dir,'realign_unwarp.mat'),'realign_unwarp')
    spm_jobman ('run',realign_unwarp);
end