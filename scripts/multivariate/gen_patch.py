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
                                 mask_img=None,
                                 empty_policy:str="ignore",
                                 n_vox:int=1,
                                 allow_overlap:bool=True,
                                 n_jobs:int=cpu_count()):
    """
    This function is adapted from `_apply_mask_and_get_affinity` from the `nilearn.maskers.nifti_spheres_masker` module
    (https://github.com/nilearn/nilearn/blob/0d379462d8f84344056308d4d096caf78954ca6d/nilearn/input_data/nifti_spheres_masker.py)
    
    Original Documentation: 
    ----------
    Get only the rows which are occupied by sphere at given seed locations and the provided radius.
    Rows are in target_affine and target_shape space.
    
    Adaptation 
    ----------
    This function is adapted to:
    1. impose minimum voxel number contraints and avoid throwing warnings when an empty sphere/insufficient voxel number is detected.
    The current script handle empty sphere or sphere with insufficient voxel number using one of the three approaches by specifying `empty_policy` argument:  
        - ``'fill'``: conduct new search with the same algorithm but incrementing searchlight sphere radius for empty spheres  
        - ``'ignore'``: return empty  
        - ``'raise'``: throw an error       

    2. add parallelization to speed up the process

    
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
        Indicates, in millimeters, the radius for the sphere around the seed.  By default `5`.

    mask_img : Niimg-like object, optional
        Mask to apply to regions before extracting signals. If niimg is None,
        mask_img is used as a reference space in which the spheres 'indices are
        placed. By default `None`.

    empty_policy: str, optional
        how to deal with empty spheres

    n_vox : int, optional
        minimum number of voxel in each searchlight sphere. By default `1`.

    allow_overlap : boolean
        If False, a ValueError is raised if VOIs overlap. By default `True`.

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

    t1 = time.time()
    print(f'image loading time: {t1-t0}')
    
    # For each seed, get coordinates of nearest voxel
    def find_ind(m_coords, nearest):
        try:
            return m_coords.index(nearest)
        except ValueError:
            return None

    def search_nearest_for_seedschunk(m_coords,seedschunk,aff):
        tmp_nearests = np.round(image.resampling.coord_transform(
            np.array(seedschunk)[:,0], np.array(seedschunk)[:,1], np.array(seedschunk)[:,2], np.linalg.inv(aff)
        )).T.astype(int)
        nearest_ = [find_ind(m_coords,tuple(nearest)) for nearest in tmp_nearests]
        return nearest_
    
    with Parallel(n_jobs=n_jobs) as parallel:
        n_splits = n_jobs
        split_idx = np.array_split(np.arange(len(seeds)), n_splits)
        seed_chunks = [np.array(seeds)[idx] for idx in split_idx]
        nearests = parallel(delayed(search_nearest_for_seedschunk)(mask_coords,sc,affine) for sc in seed_chunks)
    nearests = sum(nearests,[])
    t2 = time.time()
    print(f'get nearest time: {t2-t1}')
    
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
    t3 = time.time()
    print(f'get neighbour time: {t3-t2}')   

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        with contextlib.suppress(ValueError):
            A[i, mask_coords.index(list(map(int, seed)))] = True
    t4 = time.time()
    print(f'split spheres time: {t4-t3}')

    # Check for empty/insufficient voxel number spheres    
    sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
    redo_spheres = np.nonzero(sphere_sizes < n_vox)[0]
    
    j = 0
    if empty_policy == "raise":
        raise ValueError(f'The following spheres have less than {n_vox} voxels: {redo_spheres}')
    elif empty_policy == "ignore":
        pass
    elif empty_policy == "fill":
        #expand radius if doesn't meet voxel count criteria
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
    
    t5 = time.time()
    print(f'redo/wrapup time: {t5-t4}')

    dt = time.time()-t0
    print(f'neibourhood specification elapse time: {dt}, redo iterations = {j},  max radius = {radius}')
    return X, A, [t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,dt]


nidir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\LSA_stimuli_navigation_concatall\first\sub001'
niimg = nib.load(os.path.join(nidir,'stimuli_all.nii'))
rmask_dir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\reliability_concat\first\sub001'
process_mask_img = nib.load(os.path.join(nidir,'mask.nii'))
process_mask_data = process_mask_img.get_fdata()
process_coords = np.where(process_mask_data!=0)
process_coords = np.asarray(
    nilearn.image.coord_transform(
        process_coords[0],
        process_coords[1],
        process_coords[2],
        process_mask_img.affine
        )
    ).T
seeds=process_coords
radius = 10
allow_overlap = True
mask_img = nib.load(os.path.join(rmask_dir,'reliability_mask.nii'))
n_vox = 2

tlog = []
for njobs in [16]:
#for njobs in [18,17,16,15,14,13,12,11,10]:
    print(f'testing njobs = {njobs}')
    if njobs==16:
        X, A, t_log = _apply_mask_and_get_affinity(seeds = seeds,
                                                niimg = niimg,
                                                radius = radius,
                                                mask_img = mask_img,
                                                empty_policy="ignore",
                                                n_vox = n_vox,
                                                allow_overlap = allow_overlap,
                                                n_jobs = njobs)
    else:
        _, _, t_log = _apply_mask_and_get_affinity(seeds = seeds,
                                                niimg = niimg,
                                                radius = radius,
                                                mask_img = mask_img,
                                                empty_policy="ignore",
                                                n_vox = n_vox,
                                                allow_overlap = allow_overlap,
                                                n_jobs = njobs)
    tlog.append(t_log)

import pandas as pd
import seaborn as sns
t_arr = np.array(tlog)
col_names = ["img_load","get_nearest","get_neighbour","split_sphere","redo_wrapup","total"]
t_df = pd.DataFrame(t_arr).rename(columns=dict(zip(range(6),col_names)))
t_df["njobs"] = [18,17,16,15,14,13,12,11,10]
t_dfm = pd.melt(t_df, id_vars='njobs', value_vars=col_names)
fig = sns.relplot(data=t_dfm,x="njobs",y="value",hue="variable",kind="line")
fig