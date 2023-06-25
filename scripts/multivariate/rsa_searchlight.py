"""
This module contains classes for running searchlight RSA analysis.
The code is adapted from nilearn.decoding.searchlight module @ https://github.com/nilearn/nilearn/blob/321494420/nilearn/decoding/searchlight.py
"""
import os
import sys
import time
import json
import warnings
import numpy as np
import contextlib
from copy import deepcopy

import nibabel as nib
import nibabel.processing
from joblib import Parallel, delayed, cpu_count
from scipy.sparse import vstack, find
from multivariate.helper import compute_rdm,checkdir,scale_feature

import nilearn
from nilearn import image, masking
from nilearn._utils.niimg_conversions import (
    _safe_get_data,
    check_niimg_3d,
)
from nilearn.decoding.searchlight import GroupIterator

from sklearn import neighbors




def _apply_mask_and_get_affinity(seeds,
                                 niimg,
                                 radius:float=5,
                                 allow_overlap:bool=True,
                                 mask_img=None,
                                 n_vox:int=1):
    """
    This function is adapted from `_apply_mask_and_get_affinity` from the `nilearn.maskers.nifti_spheres_masker` module
    (https://github.com/nilearn/nilearn/blob/0d379462d8f84344056308d4d096caf78954ca6d/nilearn/input_data/nifti_spheres_masker.py)
    Custom script was added in the end to avoid throwing warnings when an empty sphere is detected. \
    Instead of throwing error for empty sphere, new search will be conducted with the same algorithm but incrementing searchlight sphere radius
    
    Get only the rows which are occupied by sphere \
    at given seed locations and the provided radius.

    Rows are in target_affine and target_shape space.

    Parameters
    ----------
    seeds : List of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as target_affine.

    niimg : 3D/4D Niimg-like object
        See :ref:`extracting_data`.
        Images to process.
        If a 3D niimg is provided, a singleton dimension will be added to
        the output to represent the single scan in the niimg.

    radius : float
        Indicates, in millimeters, the radius for the sphere around the seed.

    allow_overlap : boolean
        If False, a ValueError is raised if VOIs overlap

    mask_img : Niimg-like object, optional
        Mask to apply to regions before extracting signals. If niimg is None,
        mask_img is used as a reference space in which the spheres 'indices are
        placed.

    n_vox : int, optional
        minimum number of voxel in each searchlight sphere

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    """

    t0 = time.time()

    seeds = list(seeds)

    # Compute world coordinates of all in-mask voxels.
    if niimg is None:
        mask, affine = masking._load_mask_img(mask_img)
        # Get coordinate for all voxels inside of mask
        mask_coords = np.asarray(np.nonzero(mask)).T.tolist()
        X = None

    elif mask_img is not None:
        affine = niimg.affine
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(
            mask_img,
            target_affine=affine,
            target_shape=niimg.shape[:3],
            interpolation='nearest',
        )
        mask, _ = masking._load_mask_img(mask_img)
        mask_coords = list(zip(*np.where(mask != 0)))

        X = masking._apply_mask_fmri(niimg, mask_img)

    else:
        affine = niimg.affine
        if np.isnan(np.sum(_safe_get_data(niimg))):
            warnings.warn(
                'The imgs you have fed into fit_transform() contains NaN '
                'values which will be converted to zeroes.'
            )
            X = _safe_get_data(niimg, True).reshape([-1, niimg.shape[3]]).T
        else:
            X = _safe_get_data(niimg).reshape([-1, niimg.shape[3]]).T

        mask_coords = list(np.ndindex(niimg.shape[:3]))

    # For each seed, get coordinates of nearest voxel
    tmp_nearests = np.round(image.resampling.coord_transform(
            np.array(seeds)[:,0], np.array(seeds)[:,1], np.array(seeds)[:,2], np.linalg.inv(affine)
        )).T.astype(int)
    def find_ind(mask_coords, nearest):
        try:
            return mask_coords.index(nearest)
        except ValueError:
            return None
    nearests = [find_ind(mask_coords,tuple(nearest)) for nearest in tmp_nearests]

    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = image.resampling.coord_transform(
        mask_coords[0], mask_coords[1], mask_coords[2], affine
    )
    mask_coords = np.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue

        A[i, nearest] = True

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        with contextlib.suppress(ValueError):
            A[i, mask_coords.index(list(map(int, seed)))] = True
    sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
    redo_spheres = np.nonzero(sphere_sizes < n_vox)[0]
    
    #==================================================== customised redo neibourghood searching scripts ==================================================
    #expand radius if doesn't meet voxel count criteria
    j = 0
    while len(redo_spheres)>0:
        j+=1
        redo_seeds = list(np.array(seeds)[redo_spheres])
        redo_nearests = list(np.array(nearests)[redo_spheres])
        radius += 0.5
        redo_A = neighbors.NearestNeighbors(radius=radius).fit(mask_coords).radius_neighbors_graph(redo_seeds)
        redo_A = redo_A.tolil()
        for i, nearest in enumerate(redo_nearests):
            if nearest is None:
                continue
            redo_A[i, nearest] = True
        for i, seed in enumerate(redo_seeds):
            with contextlib.suppress(ValueError):
                redo_A[i, mask_coords.index(list(map(int, seed)))] = True
        A[redo_spheres, :] = redo_A
        sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
        redo_spheres = np.nonzero(sphere_sizes < n_vox)[0]
        
    if (not allow_overlap) and np.any(A.sum(axis=0) >= 2):
        raise ValueError('Overlap detected between spheres')
    
    dt = time.time()-t0
    print(f'neibourhood specification elapse time: {dt}, redo iterations = {j},  max radius = {radius}')
    return X, A, radius

class RSASearchLight:
    def __init__(self,
                 patternimg_paths,
                 mask_img_path:str,
                 process_mask_img_path:str=None,
                 radius:float=12.5,
                 njobs:int=1):
        """_summary_

        Parameters
        ----------
        patternimg_paths : str or list
            path to the activity pattern images. It can be path to a 4D image or paths to multiple 3D images.
        mask_img_path : str
            path to the mask image. The mask image is a boolean image specifying voxels whose signals should be included into computation
        process_mask_img_path : str, optional
            path to the process mask image. The process mask image is a boolean image specifying voxels on which searchlight analysis is performed, by default None
        radius : float, optional
            the radius of the searchlight sphere, by default 12.5
        njobs : int, optional
            number of parallel jobs, by default 1
        """
        self.mask_img         = nib.load(mask_img_path)
        process_mask_img_path = mask_img_path if process_mask_img_path is None else process_mask_img_path
        self.process_mask_img = nib.load(process_mask_img_path)
        self.radius           = radius
        self.njobs            = njobs
        if isinstance(patternimg_paths,list):
            print("concatenating images")
            self.pattern_img = nib.funcs.concat_images(patternimg_paths)
            print("finished concatenating images")
        else:
            self.pattern_img = nib.load(patternimg_paths)
        self.X, self.A = self.genPatches()
        self.neighbour_idx_lists = self.find_neighbour_idx()
        print(f"total number of voxels to perform searchlight: {len(self.neighbour_idx_lists)}")

        ## create a searchlight summary
        self.config ={"mask":mask_img_path,
                      "radius":radius,
                      "scans":patternimg_paths,
                      "njobs":njobs}

    def find_neighbour_idx(self):
        voxel_neighbours = []
        for _,row in enumerate(self.A):
            _,vidx,_ = find(row)
            voxel_neighbours.append(vidx) 
        return voxel_neighbours

    def run(self,estimator,models,modelnames,outputpath,outputregexp,verbose):
        self.estimator = estimator
        # split patches for parallelization
        group_iter = GroupIterator(len(self.neighbour_idx_lists), self.njobs)
        with Parallel(n_jobs=self.njobs) as parallel:
            results = parallel(
                delayed(self.fitPatchGroup)(
                    [self.neighbour_idx_lists[i] for i in list_i],
                    models,
                    thread_id + 1,
                    self.A.shape[0],
                    verbose,
                )
                for thread_id, list_i in enumerate(group_iter)
            )

        result = np.vstack(results)
        self.write(result,models,modelnames,outputpath,outputregexp)
        return self
    
    def write(self,result,models,modelnames,outputpath,outputregexp,ensure_finite:bool=False):
        if '.nii' not in outputregexp:
            outputregexp = f'{outputregexp}.nii'
        checkdir(outputpath)
        mean_img = nilearn.image.mean_img(self.pattern_img)
        maskdata, _ = nilearn.masking._load_mask_img(self.process_mask_img)
        for k in np.arange(np.shape(result)[1]):
            result_3D = np.full(self.process_mask_img.shape,np.nan)
            result_3D[maskdata] = result[:,k]
            if not ensure_finite:
                result_3D = result_3D.astype('float64') # make sure nan is saved as nan
            print(f"len_result = {len(result[:,k])}")
            curr_img = nilearn.image.new_img_like(mean_img, result_3D)
            curr_fn = os.path.join(outputpath,outputregexp % (k))
            nib.save(curr_img, curr_fn)
        # create a json file storing regressor information
        json_fn = os.path.join(outputpath,'searchlight.json')
        self.config.update({"estimator":str(self.estimator)})
        config = deepcopy(self.config)
        X = {"names":modelnames,
             "xX":[m.tolist() for m in models]}
        with open(os.path.join(outputpath,json_fn), "w") as outfile:
            json.dump({"config":config,"X":X}, outfile)
        
        return self 

    
    def fitPatchGroup(self,neighbour_idx_list, model,thread_id,total,verbose:bool = True):
        voxel_results = []
        t0 = time.time()
        for i,neighbour_idx in enumerate(neighbour_idx_list):
            # centering feature columns (for each voxel, demean the column corresponding to its activity)
            patternmatrix = scale_feature(self.X[:,neighbour_idx],1,False)
            neuralrdm = compute_rdm(patternmatrix,"correlation")
            # instantiate estimator for current voxel
            curr_estimator =  self.estimator(
                neuralrdm,
                model)
            # perform estimation
            voxel_results.append(curr_estimator.fit().result)
            if verbose:
                step = 10000 # print every 10000 voxels
                if  i % step == 0:
                    crlf = "\r" if total == len(neighbour_idx_list) else "\n"
                    pt = round(float(i)/len(neighbour_idx_list)*100,2)
                    dt = time.time()-t0
                    remaining = (100-pt)/max(0.01,pt)*dt
                    sys.stderr.write(
                        f"job # {thread_id}, processed {i}/{len(neighbour_idx_list)} voxels"
                        f"({pt:0.2f}%, {remaining} seconds remaining){crlf}"
                    )
        sys.stderr.write(f"job #{thread_id}, processed {len(neighbour_idx_list)} voxels in {time.time()-t0} seconds\r")
        return np.asarray(voxel_results)    

    def genPatches(self,use_parallel:bool = True,verbose:bool=False):
        print("generating searchlight patches")
        ## voxels to perform searchlight on
        process_mask_data = self.process_mask_img.get_fdata()
        process_coords = np.where(process_mask_data!=0)
        process_coords = np.asarray(
            nilearn.image.coord_transform(
                process_coords[0],
                process_coords[1],
                process_coords[2],
                self.process_mask_img.affine
                )
            ).T
        
        def get_patch_data(coords):
            """retrieve data and neibourgh for a list of searchlight patch(sphere).
            see `_apply_mask_and_get_affinity` in `nilearn` package for more details

            Parameters
            ----------
            coords : list or numpy.ndarray
                coordinate of the centre of the sphere in 3D

            Returns
            -------
            X : 2D numpy.ndarray
                activity pattern matrix from within mask voxels. shape: (number of scans, number of voxels within mask)
            A : scipy.sparse.lil_matrix
                neibourghs that should be included in each searchlight patch(sphere). shape: (number of seeds, number of voxels)
            """
            X,A,_ = _apply_mask_and_get_affinity(
            seeds  = coords,
            niimg  = self.pattern_img,
            radius = self.radius,
            allow_overlap = True,
            mask_img      = self.mask_img,# only include voxel in the mask
            n_vox = 50 # enforce a minimum of 50 voxels
            )        
            return X,A

        if use_parallel: 
            njobs = self.njobs
            split_idx = np.array_split(np.arange(len(process_coords)), njobs)
            pc_chunks = [process_coords[idx] for idx in split_idx]           
            XA_list = Parallel(n_jobs = njobs)(delayed(get_patch_data)(coords) for coords in pc_chunks)
            X = XA_list[0][0] # X is the same for each chunk
            A = vstack([l[1] for l in XA_list])
        else:
            X,A = get_patch_data(process_coords)
        A = A.tocsr()
        if verbose:
                print(f'number of voxels per sphere:{np.unique(A.sum(axis=1))}')
        print("finished generating searchlight patches") 
        return X, A 