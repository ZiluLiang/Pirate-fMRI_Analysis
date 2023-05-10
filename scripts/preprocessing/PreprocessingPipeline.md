# fMRI data Preprocessing Pipeline

## Overview
For each participants, we acquire  fmri data from 5 functional runs (EPI images), fieldmap and a high resolution T1 image.  
First, we perform auto-reorientation so that all the acquired images are rougly in the same coregister as the template image. 
Second, we calculated the voxel displacement map from fieldmap image. During realign and unwarp, we corrected signal distortion based on the calculated VDM image and realign the image to the first volume of the first run.  
Third, we coregister the T1 image to the mean EPI image of the first functional run.  
Fourth, we performed segmentation and normalization on the coregistered T1 image using the default TPM template in SPM.  
Finally, the normalization parameter estimated using the coregistered T1 image were used to normalize all the EPI images.

After preprocessing, quality checks were conducted.

## Auto-reorientation
Automatically reoriente the acquired images to the MNI space are performed before all other preprocessing steps so that we could provide a better prior for performing the segmentation and normalization. The T1 image was used to estimate a rigid transformation matrix. This matrix is then applied to the functional images and the fieldmap images of the same participants.  
[`reorient.m`](reorient.m) is called to perform this step. It is adapted from the following resources:  
[lrq3000/spm_auto_reorient_coregister](https://github.com/lrq3000/spm_auto_reorient_coregister)   
[SPMwiki: How_to_automatically_reorient_images?](https://en.wikibooks.org/wiki/SPM/How-to#How_to_automatically_reorient_images)  
[SPMwiki: How to manually change the orientation of an image?](https://en.wikibooks.org/wiki/SPM/How-to#How_to_manually_change_the_orientation_of_an_image?)
In essence, this step is performs coregistration with the template image, but only the rigid transformation part is used. To be more specific, the `auto_reorient` sub function estimates the 6 parameters of rigid transformation that makes the transformed source image as closely align with the template image as possible. This is achieved in two steps with different cost function used for estimation:  
1) affine registration using least squares (calling `spm_affreg`)
2) coregistration using normalized mutual information (calling `spm_coreg`)
In each step, only the rigid transformation matrix is used to apply transformation.
Manual inspection on reorientation quality was carried out. But no manual reorientation was performed after inspection.

## Realignment and Unwarping using fieldmap
voxel displacement map was calculated based on the reoriented fieldmap images. It is done by calling `calculatedVDM.m`(calculatedVDM.m) function and the defaults specified in `pm_defaults_Prisma_CIMCYC.m`.  The resulting vdm image was then used by the `realign_unwarp.m` to perform distortion correction. `realign_unwarp` also realigns the first scan of each run to the first scan of the first run, and realign the other scans to the first scan within each run. this is also performed on the reoriented images.


## Coregistration  
A mean EPI image of the first functional run was created after realign_unwarp step. This was used as the reference image in coregistration step. This is done by calling the `coregister.m` participants reoriented T1 image was used as the source image to coregister to this mean EPI image.

## Segmentation and Normalisation
The coregistered T1 image was used to perform segmentation and normalization based on the SPM TPM template (`segment.m`). The estimated parameters were then applied to realigned and unwarped EPI images for normalization (`normalize.m`).

## Quality checks