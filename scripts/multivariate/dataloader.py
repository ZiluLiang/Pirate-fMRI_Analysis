import os
import numpy as np
import nibabel as nib
import nibabel.processing
from nilearn.masking import apply_mask,intersect_masks,_load_mask_img
import sklearn
from nilearn.image import new_img_like
class ActivityPatternDataLoader:
    def __init__(self,data_nii_paths: list, mask_imgs=None):
        
        # validate input
        if isinstance(data_nii_paths,list):
            if not np.all([os.path.exists(x) for x in data_nii_paths]): 
                non_existing_paths = np.array(data_nii_paths)[[not os.path.exists(x) for x in data_nii_paths]]
                file_err_msg = "The following nii files are not found:\n" + ',\n'.join(non_existing_paths)
                raise LookupError(file_err_msg)
            else:
                data_img = nib.funcs.concat_images(data_nii_paths)
        else:
            data_img = nib.load(data_nii_paths)

        # combine masks
        if mask_imgs is not None:
            if isinstance(mask_imgs,list):
                loaded_maskimgs = [nib.load(mask_img) if isinstance(mask_img, str) else mask_img for mask_img in mask_imgs]
                self.mask_img = intersect_masks(loaded_maskimgs,threshold=1)
            else:
                self.mask_img = nib.load(mask_imgs) if isinstance(mask_imgs, str) else mask_imgs     
            
            X = apply_mask(data_img, self.mask_img,ensure_finite = False)
        else:
            self.maskdata = None
            X_4D = data_img.get_fdata()
            # reshape into n_condition x nvoxel 2D array
            X = np.array([X_4D[:,:,:,j].flatten() for j in range(X_4D.shape[3])])

        self.X = X
                    
    def split_data(self,groups:np.ndarray):
        splitted_X = []
        for k in np.unique(groups):            
            curr_split_X = self.X[np.where(np.array(groups)==k)]
            splitted_X.append(curr_split_X)
        return splitted_X
    
    def create_img(self,data):
        data = np.array(data)
        assert data.ndim<=2, "data should be 1d or 2d array"
        data = np.atleast_2d(data)
        assert data.shape[1]==self.X.shape[1], "data columns should correspond to masked voxels"

        data_3D_list, data_img_list = [],[]
        for data1d in data:
            data_3D = np.full(self.mask_img.shape,False)
            maskdata, _ = _load_mask_img(self.mask_img)
            data_3D[maskdata] = data1d
            data_img = new_img_like(self.mask_img, data_3D)
            data_3D_list.append(data_3D)
            data_img_list.append(data_img)

        return data_3D_list,nib.funcs.concat_images(data_img_list)