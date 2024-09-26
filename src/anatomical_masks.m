% This script generates anatomical masks in AAL3v1 parcellation using
% marsbar and puts the ROI generated using Abn.
% For definition of anatomical masks see AAL_anatomical_masks.json
% -----------------------------------------------------------------------    
% Author: Zilu Liang
clear;clc;
[directory,packages] = get_pirate_defaults(false,'directory','packages');
mask_dir  = fullfile(directory.fmri_data,'masks');   
ref_img =  fullfile(directory.fmri_data,'unsmoothedLSA','LSA_stimuli_localizer','first','sub001','mask.nii');
marsbar('on')% Start marsbar to make sure spm_get works

%% generate marsbar compatible aal3 rois
marsbar_aal3 = fullfile(mask_dir,'marsbarAAL3');
checkdir(marsbar_aal3) 
AAL3_path = fullfile(mask_dir,'AAL3v1_for_SPM12'); % the downloaded and unzipped AAL3v1 parcellations
% choose ROI_MNI_V7_1mm_list to use the higher resolution of the new parcellation results
AAL3_ROI_list = {load(fullfile(AAL3_path,'ROI_MNI_V7_1mm_list'),'ROI').ROI.Nom_L}; 
AAL3_ROI_ids = [load(fullfile(AAL3_path,'ROI_MNI_V7_1mm_list'),'ROI').ROI.ID]; 
% generate marsbar aalrois in .mat format
gen_marsbar_AALROIs(AAL3_ROI_list,AAL3_ROI_ids, ...
                 fullfile(AAL3_path,'ROI_MNI_V7_1mm.nii'), ...
                 marsbar_aal3)

% combine anatomical rois and output to the same space as participants first level mask
AAL_anat_masks = loadjson(fullfile(directory.projectdir,'src','AAL_anatomical_masks.json'));
outputdir  = fullfile(mask_dir,'anat_AAL3');
checkdir(outputdir);
cellfun(@(x) generate_anatomical_masks(marsbar_aal3,AAL_anat_masks.(x),x,outputdir,ref_img),fieldnames(AAL_anat_masks))


%% generate marsbar compatible HCP rois
marsbar_HCP = fullfile(mask_dir,'marsbarHCP');
HCP_anat_masks = loadjson(fullfile(directory.projectdir,'src','HCP-MMP_anatomical_masks.json'));
checkdir(marsbar_HCP) 

HCP_path = fullfile(mask_dir,'glasser_MNI152NLin6Asym_labels_p20'); % the downloaded and unzipped AAL3v1 parcellations
HCP_table = readtable(fullfile(mask_dir,'glasser_MNI152NLin6Asym_labels_p20','glasser_MNI152NLin6Asym_labels_p20.txt'));
atlasfname = fullfile(HCP_path,'glasser_MNI152NLin6Asym_labels_p20.nii');
% choose ROI_MNI_V7_1mm_list to use the higher resolution of the new parcellation results
HCP_ROI_list = HCP_table.label; 
HCP_ROI_ids = HCP_table.Index; 
sp = mars_space(spm_vol(ref_img));

ROIs = fieldnames(HCP_anat_masks);
for ir = 1:numel(ROIs)
    region_subregions = HCP_anat_masks.(ROIs{ir});
    sides = {'left','right'};
    side_rois = cell(1,numel(sides));
    for k = 1:numel(sides)
        s = sides{k};
        sidesubrois = cell(1,numel(region_subregions));
        for j = 1:numel(region_subregions)
            curr_roiname = sprintf("%s_%s_ROI",upper(s(1)),region_subregions{j});       
            idx = HCP_ROI_ids(strcmp(HCP_ROI_list,curr_roiname));
            sidesubrois{k} = maroi_image(struct('vol', spm_vol(atlasfname), ...
                                                'func',sprintf('img == %d',idx),...
                                                'descrip', curr_roiname,'label', curr_roiname,...
                                                'binarize',1));
            if j == 1
                side_rois{k} = sidesubrois{k};
            else
                side_rois{k} =  domaths('or',side_rois{k},sidesubrois{k});
            end
            side_rois{k} = label(side_rois{k}, sprintf("%s_%s",ROIs{ir},s));
            side_rois{k} = descrip(side_rois{k},sprintf("%s_%s",ROIs{ir},s));
        end
        mars_display_roi('display', side_rois{k})
        op_path = fullfile(marsbar_HCP,sprintf("%s_%s.nii",ROIs{ir},s));
        save_as_image(side_rois{k}, char(op_path),sp);
    end
    
    bilat_roi = domaths('or',side_rois{1},side_rois{2});
    op_path = fullfile(marsbar_HCP,sprintf('%s_bilateral.nii',ROIs{ir}));
    save_as_image(bilat_roi, char(op_path),sp);


end


%% generate marsbar compatible aal3 rois
%% ========================================================================
%          function: generate_anatomical_masks
%  ========================================================================
%  generate anatomical mask in the same space as reference image
%  INPUT:
%    - marsbar_aal3: directory to marsbar generated anatomical ROI
%    definitions in .mat format. Each .mat file corresponds to one anatomical
%    label in the aal3v1 parcellation.
%    - structures: which anatomical region is used to generate the mask.
%    the name must be from the AAL3v1 label list
%    - maskname: the name of the output mask images.
%    - outputdir: where to save the output mask images.
%    - ref_img:  the reference image. output mask images will be generated
%    in the same space as this image
%
% This function will create:
%    three masks (.nii) and definitions(.mat) of one anatomical region:
%      two unilateral masks (maskname_left.nii/.mat, maskname_right.nii/.mat),
%      one bilateral mask (maskname_bilateral.nii/.mat)

function generate_anatomical_masks(marsbar_aal3,structures,maskname,outputdir,ref_img)
    marsbar('on')% Start marsbar to make sure spm_get works
    spm('defaults', 'fmri');
    
    % pick the region to combine into roi mask
    if ischar(structures)
        structures = cellstr(structures);
    end

    % check for unilateral masks
    rois_L = cellfun(@(x) fullfile(marsbar_aal3,[x,'_L.mat']),structures,'uni',0);
    rois_R = cellfun(@(x) fullfile(marsbar_aal3,[x,'_R.mat']),structures,'uni',0);
    if all(cellfun(@(x) exist(x,'file'),rois_L)) && all(cellfun(@(x) exist(x,'file'),rois_R))
        rois = cell2struct({rois_L,rois_R,[rois_L,rois_R]},...
                       {'left','right','bilateral'},...
                       2);
    else
        rois_B = cellfun(@(x) fullfile(marsbar_aal3,[x,'.mat']),structures,'uni',0);
        rois = cell2struct({rois_B},...
                       {'bilateral'},...
                       2);
    end
    
    LR = fieldnames(rois);
    for j = 1:numel(LR)
        o_cell_arr = maroi(rois.(LR{j}));
        
        % Combine for trimmed ROI
        comb_roi = o_cell_arr{1};
        for k = 2:numel(o_cell_arr)
            comb_roi = domaths('or',comb_roi,o_cell_arr{k});
        end
        comb_roi = label(comb_roi, [maskname,'_',LR{j}]);
        comb_roi = descrip(comb_roi,[maskname,'_',LR{j}]);
        
        
        %comb_roi = rebase(comb_roi, sp);
        mars_display_roi('display', comb_roi)
        
        
        % save ROI to MarsBaR ROI file, in current directory, just to show how
        saveroi(comb_roi, fullfile(outputdir,[maskname,'_',LR{j},'.mat']));
        
        % Save as image    
        sp = mars_space(spm_vol(ref_img));
        save_as_image(comb_roi, fullfile(outputdir,[maskname,'_',LR{j},'.nii']),sp);
    end
end

%% ========================================================================
%          function: gen_marsbar_ROIs
%  ========================================================================
% This script is an adaptation from MarsBaR repo
% (https://github.com/marsbar-toolbox/marsbar-aal-rois/releases)
% to generate MarsBaR-compatiblev anatomical mask based on AAL3v1 parcellations
% or parcellations provided in the jubrain-anatomy toobox 
% ------------------------------------------------------------------
% Original Documentation:
% Script to make AAL ROIs for MarsBaR
% You will need SPM(99 or 2), and MarsBaR, on the matlab path
% ------------------------------------------------------------------
% adapted by Zilu Liang
function gen_marsbar_AALROIs(parcellation_list,parcellation_ids,parcellation_nii,save_roi_path)
    if nargin<3, parcellation_nii = spm_get(-1, '', 'Select path containing parcellation nii file'); end 
    if nargin<4, save_roi_path = spm_get(-1, '', 'Select path for saving MarsBaR ROI files'); end    
    
    
    spm('defaults', 'fmri');
    
    % ROI image
    vol = spm_vol(parcellation_nii);
    
    for r = 1:length(parcellation_list)
        nom = parcellation_list{r};
        func = sprintf('img == %d', parcellation_ids(r)); 
        o = maroi_image(struct('vol', vol, 'binarize',1,...
			                 'func', func, 'descrip', nom, ...
			                 'label', nom));
        saveroi(maroi_matrix(o), fullfile(save_roi_path,[nom '.mat']));
    end
end