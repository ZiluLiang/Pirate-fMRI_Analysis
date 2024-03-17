function reorient(subimg_dir)
% VERY VERY ROUGHLY automatically reorient T1, fieldmap and functional 
% images according to the AC-PC axis and setting the origin to AC. 
%
% INPUT:
%  - subimg_dir: directory to participant's preprocessing fmri images
% 
% This will create the following files:
%  o*.nii(reoriented images)
%
% Notes:
% 1. for the current preprocessing pipeline, this is done in order to
% provide better prior for normalization, so the template is in line with
% the one used for normalization.
%
% 2. for this script, this is done by performing rigid body 
% transformation on T1,functional, and fieldmap images according to a 
% specified template. The same set of transformation parameter is applied 
% to all the above modalities to make sure everyting moves together. The
% function used for deriving such parameter is adapted from the github repo
% spm_auto_reorient_coregister RRID: SCR_017281. https://github.com/lrq3000/spm_auto_reorient_coregister
%
% 3. for this script, the transformation parameter is estimated based on T1 
% and a specified template. Note that this may not yield satisfactory result
% if T1 and the functional and/or fieldmap scans are not acquired on the
% same day. If this is the case, using the first scan of the functional
% image to estimate the transformation parameter may be a better choice.
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    [filepattern,directory] = get_pirate_defaults(false,'filepattern','directory');    

    % find template image, if specifed template is not found, use spm's
    % avg152 instead
    if isfile(directory.mni_template)
        template_img = directory.mni_template;
    else
        template_img = fullfile(spm('Dir'),'canonical','avg152T1.nii');
    end
    
    % find paths of T1, functional and fieldmap images    
    nii_files   = structfun(@(scantype) ...
                              structfun(@(pattern) cellstr(spm_select('FPList', subimg_dir, [pattern,'.*\.nii'])),...
                                         scantype,'UniformOutput',false),...
                           filepattern.raw,...
                           'UniformOutput',false);
    
    %flatten the cell specifying paths to functional and fieldmap
    func_fmap = cellfun(@(f) struct2cell(nii_files.(f)),{'fieldmap','functional'},'uni',0);
    func_fmap = cellfun(@(x) cat(1,x{:}),func_fmap,'uni',0);
    func_fmap = cat(1,func_fmap{:});    
    
    auto_reorient(nii_files.anatomical.T1{1},template_img,func_fmap{:})
end

function auto_reorient(source_img, template_img, varargin)
% automatically reorient images according to the AC-PC axis and
% setting the origin to AC. This is done by performing rigid body transformation
% according to a specified template.
% Arguments:
% - source_img: the image to be aligned, one file name
% - template_img: the template image that the source needs to align with
% - varargin: other images that the parameter estimated using the source image
%              will be applied to for reorientation.
% --------------------------------------------------------------------------
% Copyright: The code is originally from https://github.com/lrq3000/spm_auto_reorient_coregister/blob/master/spm_auto_reorient.m
% adapted by Zilu Liang, 2023

    if ~ischar(source_img)
        [source_img,selected] = spm_select(1,'image','Please select one volume of image as the source image');
        if ~selected
            error('Source image not specified, unable to proceed')
        end
    end

    other_imgs = varargin;

    % create a copy of the source image so that transformation results is saved
    % to another file without overwriting the original file
    [rsrc,parent_dir] = create_reorient_copy(source_img);

    %% transformation 1: Affine Coregistration
    % smooth the source img to make estimation easier
    tmpsrc_img = fullfile(parent_dir,'tmp.nii');
    spm_smooth(rsrc,tmpsrc_img,[12 12 12]);

    flag_aff = struct('regtype','rigid');
    vg = spm_vol(template_img); % Vector of template volumes
    vf = spm_vol(tmpsrc_img);   % Source volume, choose the smoothed temporary image
    M_aff  = spm_affreg(vg,vf,flag_aff);%estimate affine transform parameters, such that voxels in VF map to those in VG by VG.mat\M*VF.mat
    [u,~,v] = svd(M_aff(1:3,1:3));
    M_aff(1:3,1:3) = u*v';

    % apply it on source image
    N  = nifti(rsrc);
    N.mat = M_aff*N.mat;
    create(N);% Save the transform into nifti file headers

    %% transformation 2: Mutual Information Coregistration
    % for more detailed explanation on estimation parameter choice,check out documentation at
    % https://github.com/lrq3000/spm_auto_reorient_coregister/blob/master/spm_auto_reorient.m
    flag_mi = struct('cost_fun','nmi',...
                     'tol',[0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]);
    vg = spm_vol(template_img); % Vector of template volumes
    vf = spm_vol(rsrc); % Source volume
    M_mi = spm_coreg(vg,vf,flag_mi);

    % apply it on source image
    N = nifti(rsrc);
    N.mat = spm_matrix(M_mi)\N.mat;
    create(N);% Save the transform into nifti file headers

    %% erase the smoothed temporary image from disk
    spm_unlink(tmpsrc_img);

    %% apply transformation to other image files.
    for j = 1:numel(other_imgs)
        r_othersrc = create_reorient_copy(other_imgs{j});
        No  = nifti(r_othersrc);
        No.mat = M_aff*No.mat;
        No.mat = spm_matrix(M_mi)\No.mat;    
        create(No);% Save the transform into nifti file headers
    end
end

function [rsrc,parent_dir] = create_reorient_copy(src)
    % create a copy of the source image so that transformation results is 
    % saved to another file with the prefix 't' instead of overwriting the
    % original image
    % INPUT: full path to the source image to be reoriented
    % OUTPUT: full path to the copy of source image to be oriented,
    %         parent directory where the file is stored
    [parent_dir,src_fn,ext] = fileparts(src);
    rsrc = fullfile(parent_dir,['o',src_fn,ext]);
    copyfile(src,rsrc)
end