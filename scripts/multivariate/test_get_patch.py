"""Transformer for computing seeds signals.

Mask nifti images by spherical volumes for seed-region analyses
"""


import contextlib
import warnings

import numpy as np
from nilearn import image, masking
from nilearn._utils.niimg_conversions import (
    _safe_get_data,
    check_niimg_3d,
)
from sklearn import neighbors

def _apply_mask_and_get_affinity(seeds,
                                 niimg,
                                 radius,
                                 allow_overlap,
                                 mask_img=None,
                                 n_vox=1):
    """Get only the rows which are occupied by sphere \
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

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    """



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
    print('get coordinates of nearest voxel')
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(image.resampling.coord_transform(
            sx, sy, sz, np.linalg.inv(affine)
        ))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)

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
    print('Include the voxel containing the seed itself if not masked')
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        with contextlib.suppress(ValueError):
            A[i, mask_coords.index(list(map(int, seed)))] = True
    sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
    redo_spheres = np.nonzero(sphere_sizes < n_vox)[0]
    print(f'redo spheres = {redo_spheres}')

    #expand radius if doesn't meet voxel count criteria
    j = 0
    while len(redo_spheres)>0:
        j+=1
        print(f'redo loop: {j}')
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
        print(f'redo spheres = {redo_spheres}')

    if (not allow_overlap) and np.any(A.sum(axis=0) >= 2):
        raise ValueError('Overlap detected between spheres')

    return X, A, radius

############################debugging######################################
import nilearn
import nibabel as nib
import time
niimg = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\LSA_stimuli_navigation\first\sub002\stimuli_mu.nii'
niimg = nib.load(niimg)
process_mask = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\LSA_stimuli_navigation\first\sub002\mask.nii'
process_mask = nib.load(process_mask)
process_mask_data = process_mask.get_fdata()
coords = np.where(process_mask_data!=0)
coords = np.asarray(
            nilearn.image.coord_transform(
                coords[0],
                coords[1],
                coords[2],
                process_mask.affine
                )
            ).T
#seeds = coords[[2000,3874,3905,6217]]
#seeds = coords[9000:10000]
seeds = coords
radius = 12.5
allow_overlap = True
mask_img = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\LSA_stimuli_navigation\first\sub002\reliability_mask.nii'
mask_img = nib.load(mask_img)
n_vox = 50
t0 = time.time()
X, A, radius = _apply_mask_and_get_affinity(seeds,
                                 niimg,
                                 radius,
                                 allow_overlap,
                                 mask_img=None,
                                 n_vox=1)
dt = time.time()-t0
print(f'time elapsed: {dt}, max radius = {radius}')
############################debugging######################################