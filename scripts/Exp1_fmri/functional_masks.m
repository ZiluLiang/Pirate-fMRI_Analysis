% This script combines unilateral functional masks into bilateral ones
% -----------------------------------------------------------------------    
% Author: Zilu Liang

clear;clc;
[directory,packages] = get_pirate_defaults(false,'directory','packages');
mask_dir  = fullfile(directory.fmri_data,'masks');   
ref_img =  fullfile(directory.fmri_data,'unsmoothedLSA','LSA_stimuli_localizer','first','sub001','mask.nii');

%%
func_mask_dir = fullfile(directory.fmri_data,"unsmoothedLSA","rsa_searchlight","fourruns_noselection_mvnn_averageall",...
                "regression\compete_featurecartesian_combinexy_teststim", ...
                "bspmclustermask");
ROIs = {"testgtlocParietalSup","testgtlocTPMid"};

for ir = 1:numel(ROIs)
    unilat_mask_files = spm_select('FPList',func_mask_dir,ROIs{ir});
    cellfun(@(x) sprintf("individual mask, total voxel = %d", spm_summarise(char(x),'all','sum')), cellstr(unilat_mask_files),'uni',0)
    opname = char(fullfile(func_mask_dir,sprintf('%s_bilateral.nii',ROIs{ir})));
    Vo = spm_imcalc(unilat_mask_files, opname, '(i1 + i2)>0',{0,0,0,4});
    sprintf("%s mask, total voxel = %d", ROIs{ir}, spm_summarise(Vo,'all','sum'))
end