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
Automatically reoriente the acquired images to the MNI space before all other preprocessing steps so that we could provide a better prior for performing the segmentation and normalization. The T1 image is used to estimate a rigid transformation matrix. This matrix is then applied to the functional images and the fieldmap images of the same participants.  [`reorient.m`](reorient.m) is called to perform this step. It is adapted from the following resources:  
[lrq3000/spm_auto_reorient_coregister](https://github.com/lrq3000/spm_auto_reorient_coregister)   
[SPMwiki: How_to_automatically_reorient_images?](https://en.wikibooks.org/wiki/SPM/How-to#How_to_automatically_reorient_images)  
[SPMwiki: How to manually change the orientation of an image?](https://en.wikibooks.org/wiki/SPM/How-to#How_to_manually_change_the_orientation_of_an_image?)  

In essence, this step is performs coregistration with the template image, but only the rigid transformation part is used. To be more specific, the [`auto_reorient`](reorient.m#L43)) sub function estimates the 6 parameters of rigid transformation that makes the transformed source image as closely align with the template image as possible. This is achieved in two steps with different cost function used for estimation:  
1) affine registration using least squares (calling `spm_affreg`)
2) coregistration using normalized mutual information (calling `spm_coreg`)  

In each step, only the rigid transformation matrix is used to apply transformation.  

Manual inspection on reorientation quality was carried out. But no manual reorientation was performed after inspection.

## Realignment and Unwarping using fieldmap
Air-tissue interfaces lead to inhomogeneity of the $B_0$ field (the main magnetic field). This in turn leads to two forms of artifacts near air-tissue interfaces: signal dropout and distortion. Distortion is usually corrected by the fieldmap acquired. But signal loss cannot be recovered.
To correct for signal distortion, Voxel displacement map is first calculated based on the reoriented fieldmap images. This is done by calling [`calculateVDM`](calculateVDM.m) function and the defaults specified in [`pm_defaults_Prisma_CIMCYC`](pm_defaults_Prisma_CIMCYC.m). The resulting vdm image was then used by the [`realign_unwarp`](realign_unwarp.m) to perform distortion correction.
For detailed explanation on distortion correction and how to set the pm_defaults files see:
1. [Handbook of functional mri data analysis](https://www.cambridge.org/core/books/handbook-of-functional-mri-data-analysis/8EDF966C65811FCCC306F7C916228529) Chapter 3 section 3.4   
2. [Acquiring and using field maps lewis center of neuro imaging](https://lcni.uoregon.edu/kb-articles/kb-0003)


[`realign_unwarp`](realign_unwarp.m) also realigns the first scan of each run to the first scan of the first run, and realign the other scans to the first scan within each run. This is also performed on the reoriented images. One ``rp_*.txt`` file is generated for each run, this can be used as estimates of head motion in the scanner (for first level nuisance regressors or quality inspection). A mean EPI image of the first functional run (``mean*.nii``) is generated as well.


## Coregistration  
 During coregistration, the anatomical image is coregistered to the epi image so that later the normalization parameter based on anatomical image can be applied to the EPI images. This is done by calling the [`coregister`](coregister.m). Participants' reoriented T1 image was used as the source image while the mean EPI image of the first functional run (``mean*.nii``) was used as reference image.

## Segmentation and Normalisation
In SPM12, segmentation and normalization in unified into one module. This step segments and normalises the coregistered T1 image based on the SPM TPM template (calling [`segment`](segment.m)). The estimated parameters were then applied to realigned and unwarped EPI images for normalization (calling [`normalise`](normalise.m)). Note that in [`normalise`](normalise.m#L14) the size of bounding box was set to be the same size as the template image, and that B-splinethe interpolation with 7-degree splines is used to achieve higher quality.

## Smoothing
Following suggestions in [Handbook of functional MRI data analysis Chapter 3.7.1](https://www.cambridge.org/core/books/handbook-of-functional-mri-data-analysis/preprocessing-fmri-data/76A784C9F6369B1EA1DFC89EF394251C), normalised nii files are smoothed using fwhm that is twice the voxel size (`[5,5,5]`).

## Quality checks
Some handy functions are provided in the [`qualitycheck`](/scripts/qualitycheck) directory to perform manual inspections of preprocessing quality. Currently, the following checks are performed in [`quality_check`](/scripts/quality_check.m). The results are logged manually in [QualityCheckLogBook](https://unioxfordnexus-my.sharepoint.com/:x:/r/personal/sedm6713_ox_ac_uk/Documents/Project/pirate_fmri/Analysis/data/fmri/qualitycheck/QualityCheckLogBook.xlsx?d=w4d93d7284861418bbad93c525fa01b30&csf=1&web=1&e=YjFIZj) ([local copy](/data/fmri/qualitycheck/QualityCheckLogBook.xlsx))
1. check quality of signal distortion correction
   A quality check for this process is to compare the image before and after distortion correction to see if there are less signal loss near air-tissue interfaces, which include regions such as orbitalfrontal cortex, anterial perfrontal cortex as well as lateral temporal lobe.  
   This can be done after VDM calculation is finished. VDM calculation will create an unwarped (distortion corrected) version of the first epi (first volume of the first task run). By putting unwarped epi, original epi and t1 image side by side, and checking different regions, it will be clear whether the unwarped image is better than the original epi.   
   This can also be done after realign_unwarp is finish. First select random volume from an EPI timeseries and put the unwarped epi, original epi and t1 side by side to see if unwared version is better than original.     
2. check quality of spatial coregistration
   checks for spatial coregistration quality can be performed after: 1) reorientaion, 2) t1 to epi coregistration, 3) normalization.
   Usually, coregistered source image and reference image are put side by side to see if there are big deviations 
3. check head motion
   Head motion estimats are extracted from `rp_*.txt` files and plotted as line graph. An alternative is to view the animation of 4D series in mricroGL to see if there are sudden shifts of volumes. Both can be done by calling [`check_head_motion`](/scripts/qualitycheck/check_head_motion.m)
4. calculate tSNR
   Following the definition of [Triantafyllou et al., (2005)](https://doi.org/10.1016/j.neuroimage.2005.01.007), [`calculate_snr.m`](/scripts/qualitycheck/calculate_snr.m) calculate voxel-wise tsnr for a given timeseries (a 4D nii file), it can further calculate the mean tsnr across the whole brain or within a masked region.