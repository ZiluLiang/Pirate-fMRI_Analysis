% This script generates anatomical masks in AAL3v1 parcellation using
% marsbar.
% For definition of anatomical masks see anatomical_masks.json
% -----------------------------------------------------------------------    
% Author: Zilu Liang

directory = get_pirate_defaults(false,'directory');
mask_dir  = fullfile(directory.projectdir,'data','fmri','masks');   
marsbar_aal3 = fullfile(mask_dir,'marsbarAAL3');

%% generate marsbar compatible aal3 rois
checkdir(marsbar_aal3) 
AAL3_path = fullfile(mask_dir,'AAL3'); % the downloaded and unzipped AAL3v1 parcellations
gen_marsbar_AAL(AAL3_path,marsbar_aal3,'ROI_MNI_V7_1mm')

%% combine anatomical rois and output to  the same space as participants first level mask
ref_img = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\LSA_stimuli_localizer\first\sub001\mask.nii';
anat_masks = loadjson(fullfile(directory.projectdir,'scripts','anatomical_masks.json'));
outputdir  = fullfile(mask_dir,'anat');
checkdir(outputdir);
cellfun(@(x) generate_anatomical_masks(marsbar_aal3,anat_masks.(x),x,outputdir,ref_img),fieldnames(anat_masks))

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
        cellstr(spm_select('FPList',marsbar_aal3,structures))
    elseif iscell(structures)
        rois_L = cellfun(@(x) fullfile(marsbar_aal3,[x,'_L.mat']),structures,'uni',0);
        rois_R = cellfun(@(x) fullfile(marsbar_aal3,[x,'_R.mat']),structures,'uni',0);
    end
    rois = cell2struct({rois_L,rois_R,[rois_L,rois_R]},...
                       {'left','right','bilateral'},...
                       2);
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
%          function: gen_marsbar_AAL
%  ========================================================================
% This script is a slight adaptation from MarsBaR repo
% (https://github.com/marsbar-toolbox/marsbar-aal-rois/releases)
% to generate MarsBaR-compatiblev anatomical mask based on AAL3v1 parcellations
% 
% ------------------------------------------------------------------
% Original Documentation:
% Script to make AAL ROIs for MarsBaR
% You will need SPM(99 or 2), and MarsBaR, on the matlab path
% ------------------------------------------------------------------
% adapted by Zilu Liang
function gen_marsbar_AAL(aal_path,roi_path,aal_root)
    if nargin<1, aal_path = spm_get(-1, '', 'Select path containing AAL files'); end
    if nargin<2, roi_path = spm_get(-1, '', 'Select path for MarsBaR ROI files'); end    
    if nargin<3, aal_root = 'ROI_MNI_V7_1mm'; end % use the higher resolution of the new parcellation results
    
    marsbar('on')% Start marsbar to make sure spm_get works
    spm('defaults', 'fmri');
    
    % ROI names
    load(fullfile(aal_path, [aal_root '_List']));
    % ROI image
    img = fullfile(aal_path, [aal_root '.nii']);
    
    % Make ROIs
    vol = spm_vol(img);
    for r = 1:length(ROI)
      nom = ROI(r).Nom_L;
      func = sprintf('img == %d', ROI(r).ID); 
      o = maroi_image(struct('vol', vol, 'binarize',1,...
			     'func', func, 'descrip', nom, ...
			     'label', nom));
      saveroi(maroi_matrix(o), fullfile(roi_path,[nom '.mat']));
    end
end