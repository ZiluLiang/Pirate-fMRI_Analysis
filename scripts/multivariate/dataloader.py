import os
import numpy
import nibabel as nib
import nibabel.processing
from nilearn.masking import apply_mask,intersect_masks,_load_mask_img
import sklearn
from nilearn.image import new_img_like

def _check_single_image(image):
        if isinstance(image,str):
            if not os.path.exists(image):
                raise LookupError("The following nii files are not found:\n" + image)
            else:
                return nib.load(image)
        elif isinstance(image,nibabel.spatialimages.SpatialImage):
            return image
        else:
            raise ValueError("input must be a loaded nibabel image or path to image")
        

def _check_and_load_images(imgs,mode:str="pass",intersect_threshold=1):
        if not isinstance(imgs,list):
            imgs = [imgs]
        image_list = [_check_single_image(img) for img in imgs]

        if len(image_list)>1:
                if mode == "concatenate":
                        loaded_image = nib.funcs.concat_images(image_list,axis=3) # concatenate into 4D image
                elif mode == "intersect":
                        loaded_image = intersect_masks(image_list,threshold=intersect_threshold) # compute intersection of the masks
                elif mode == "pass":
                        loaded_image = image_list
        else:
                loaded_image = image_list[0]

        return loaded_image

class ActivityPatternDataLoader:
    def __init__(self,data_nii_paths: list, mask_imgs=None):
        """_summary_

        Parameters
        ----------
        data_nii_paths : str or list
            path or a list of paths to nii file containing the data to be loaded. If a list of paths is provided, nii files will be concatenated along the 3rd dimension (as 4D nii files)
        mask_imgs : _type_, optional
            path or a list of paths to mask image(s). 
            If None, data from all voxels in the 4D nii image will be loaded, by default None
            If a single mask is provided,  only voxels in mask will be included
            If a list of masks is provided,  only voxels lying at the intersection of the masks will be included            

        Raises
        ------
        LookupError
            if any of the nii image files are not found, will throw error
        """
        
        # load data image
        data_img = _check_and_load_images(imgs=data_nii_paths, mode="concatenate")        

        # obtain masks
        if mask_imgs is not None:
            self.mask_img = _check_and_load_images(imgs=mask_imgs, mode="intersect")            
            X = apply_mask(data_img, self.mask_img,ensure_finite = False)
        else:
            self.maskdata = None
            X_4D = data_img.get_fdata()
            # reshape into n_condition x n_voxel 2D array
            X = numpy.array([X_4D[:,:,:,j].flatten() for j in range(X_4D.shape[3])])

        self.X = X
    
    def split_data(self,groups:numpy.ndarray):
        """split activity pattern matrix row-wise into submatrices by groups

        Parameters
        ----------
        groups : numpy.ndarray
            group indices array with `numpy.size(groups) = self.X.shape[0]`

        Returns
        -------
        list
            list of submatrices
        """
        assert numpy.size(groups) == self.X.shape[0], "number of elements in group array must equal to number of rows in self.X"
        splitted_X = []
        for k in numpy.unique(groups):            
            curr_split_X = self.X[numpy.where(numpy.array(groups)==k)]
            splitted_X.append(curr_split_X)
        return splitted_X

    def create_img(self,data:numpy.ndarray,ensure_finite:bool=False):
        """create a nibabel image object to store data

        Parameters
        ----------
        data : numpy.ndarray
            a 1d or 2d data array to be saved in nii image. 
            If data only has one dimension, a 3D image will be created. data array must contain same number of data points as the number of masked voxels.
            If data has two dimensions, a 4D image will be created. The columns of data will be mapped onto masked voxels, and each row specifies one volume of the 4D image.
        ensure_finite : bool, optional
            whether or not to replace nans. If True, nans will be saved as zero. If false, will keep as nan. by default False

        Returns
        -------
        data_3D_list
            list containing data reshaped into same dimension as the mask
        img
            4D or 3D nibabel image
        """
        data = numpy.array(data)
        assert data.ndim<=2, "data should be 1d or 2d array"
        data = numpy.atleast_2d(data)
        assert data.shape[1]==self.X.shape[1], "data columns should correspond to masked voxels"

        data_3D_list, data_img_list = [],[]
        for data1d in data:
            data_3D = numpy.full(self.mask_img.shape,numpy.nan)
            maskdata, _ = _load_mask_img(self.mask_img)
            data_3D[maskdata] = data1d
            if not ensure_finite:
                data_3D = data_3D.astype('float64') # make sure nan is saved as nan
            else:
                data_3D[numpy.isnan(data_3D)] = 0 # replace nan with 0
            data_img = new_img_like(self.mask_img, data_3D)
            data_3D_list.append(data_3D)
            data_img_list.append(data_img)

        if len(data_img_list) == 1: # do not save as 4D if it is not 4D
            img = data_img_list[0]
        else:
            img = nib.funcs.concat_images(data_img_list)
        return data_3D_list, img