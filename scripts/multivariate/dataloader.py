import os
import numpy as np
import nibabel as nib
import nibabel.processing
from nilearn.masking import apply_mask,intersect_masks
import sklearn
from nilearn.image import math_img
class ActivityPatternDataLoader:
    def __init__(self,data_nii_paths: list, conditions: list, condition_names: list = None, mask_imgs=None, dropNA_flag = True):
        
        # validate input
        if not np.all([os.path.exists(x) for x in data_nii_paths]): 
            non_existing_paths = np.array(data_nii_paths)[[not os.path.exists(x) for x in data_nii_paths]]
            file_err_msg = "The following nii files are not found:\n" + ',\n'.join(non_existing_paths)
            raise Exception(file_err_msg)
        
        if not len(conditions) == len(data_nii_paths):
            size_err_msg = "The size of data paths must match the size of conditions"
            raise Exception(size_err_msg) 
        
        #extract activity pattern for each condition
        data_imgs = [nib.load(data_nii_path) for data_nii_path in data_nii_paths]
        def load_singlemask(mask_img,ref_maskimg):
            if isinstance(mask_img, str):
                mask_img = nib.load(mask_img)
            elif isinstance(mask_img, np.ndarray):
                reshapeimg = np.reshape(mask_img,data_imgs[0].shape).astype(np.int8)
                mask_img = nib.Nifti1Image(reshapeimg, data_imgs[0].affine,data_imgs[0].header)
            else:
                mask_img = mask_img
            resampled_maskimg = nib.processing.resample_from_to(mask_img, ref_maskimg) 
            return resampled_maskimg
        # combine masks
        if mask_imgs is not None:
            if isinstance(mask_imgs,list):
                loaded_imgs = [load_singlemask(mask_img,data_imgs[0]) for mask_img in mask_imgs]
                mask_img = intersect_masks(loaded_imgs,threshold=1)
            else:
                mask_img = load_singlemask(mask_imgs,data_imgs[0])       
            
            X = apply_mask(data_imgs, mask_img,ensure_finite = False)
        else:
            X = np.array([img.get_fdata().flatten() for img in data_imgs])
        
        if dropNA_flag:
            X_dropna,_ = self.drop_na(X)
            self.X = X_dropna
        else:
            self.X = X
        self.Y = np.array(conditions)

        # standardize the activity pattern matrix
        mu  = np.nanmean(self.X)
        std = np.nanstd(self.X)
        self.zX = (self.X - mu)/std

        # compute RDM
        self.RDM = self.compute_rdm(self.zX)
    
    def drop_na(self,pattern_matrix=None):
        if pattern_matrix is not None:
            X = pattern_matrix
        else:
            X = self.X                                  
        na_filters = np.all([~np.isnan(X[j,:]) for j in range(np.shape(X)[0])],0)
        X_drop_na = X[:,na_filters]
        return (X_drop_na,na_filters)
        
    def split_data(self,splits:np.ndarray):
        if not np.size(splits) == np.size(self.Y):
            raise Exception("the splits must be the same size as the conditions")
        
        splitted_X = []
        splitted_Y = []
        for k in np.unique(splits):            
            curr_split_X = self.X[np.where(np.array(splits)==k)]
            curr_split_Y = self.Y[np.where(np.array(splits)==k)]
            splitted_X.append(curr_split_X)
            splitted_Y.append(curr_split_Y)
        splitted_X = np.array(splitted_X)
        splitted_Y = np.array(splitted_Y)
        
        return (splitted_X,splitted_Y)

    def compute_rdm(self,pattern_matrix):
        X_drop_na,_ = self.drop_na(pattern_matrix)
        rdm = sklearn.metrics.pairwise_distances(X_drop_na, metric='correlation')
        return rdm