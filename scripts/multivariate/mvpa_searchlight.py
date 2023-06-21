"""
This module contains classes for running searchlight RSA and decoding analysis.
The code is adapted from nilearn.decoding.searchlight module.
"""
import os
import sys
import time
import json
import numpy as np
import nibabel as nib
import nibabel.processing
from copy import deepcopy
from joblib import Parallel, delayed, cpu_count
from scipy.sparse import vstack, find
from multivariate.helper import compute_rdm,checkdir

import nilearn
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.decoding.searchlight import GroupIterator

class MVPASearchLight:
    def __init__(self,patternimg_paths,mask_img_path,radius,
                 estimator,njobs:int=1):
        self.mask_img    = nib.load(mask_img_path)
        self.radius      = radius
        self.estimator   = estimator
        self.njobs       = njobs
        print("concatenating images")
        self.pattern_img = nib.funcs.concat_images(patternimg_paths)
        print("finished concatenating images")
        self.X, self.A = self.genPatches(self.pattern_img)
        self.neighbour_idx_lists = self.find_neighbour_idx()

        ## create a searchlight summary
        self.config ={"mask":mask_img_path,
                      "radius":radius,
                      "estimator":str(estimator),
                      "scans":patternimg_paths,
                      "njobs":njobs}

    def find_neighbour_idx(self):
        voxel_neighbours = []
        for _,row in enumerate(self.A):
            _,vidx,_ = find(row)
            voxel_neighbours.append(vidx) 
        return voxel_neighbours

    def run(self,models,modelnames,outputpath,outputregexp,verbose):

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
    
    def write(self,result,models,modelnames,outputpath,outputregexp):
        if '.nii' not in outputregexp:
            outputregexp = f'{outputregexp}.nii'
        checkdir(outputpath)
        maskdata, _ = nilearn.masking._load_mask_img(self.mask_img)
        for k in np.arange(np.shape(result)[1]):
            result_3D = np.zeros(self.mask_img.shape)
            result_3D[maskdata] = result[:,k]
            curr_img = nilearn.image.new_img_like(self.mask_img, result_3D)
            curr_fn = os.path.join(outputpath,outputregexp % (k))
            nib.save(curr_img, curr_fn)
        # create a json file storing regressor information
        json_fn = os.path.join(outputpath,'searchlight.json')
        config = deepcopy(self.config)
        X = {"names":modelnames,
             "xX":[m.tolist() for m in models]}
        with open(os.path.join(outputpath,json_fn), "w") as outfile:
            json.dump({"config":config,"X":X}, outfile)

    
    def fitPatchGroup(self,neighbour_idx_list, model,thread_id,total,verbose:bool = True):
        voxel_results = []
        t0 = time.time()
        for i,neighbour_idx in enumerate(neighbour_idx_list):
            # instantiate estimator for current voxel
            neuralrdm = compute_rdm(self.X[:,neighbour_idx],"correlation")
            curr_estimator =  self.estimator(
                neuralrdm,
                model)
            # perform estimation
            voxel_results.append(curr_estimator.fit().result)
            if verbose:
                step = 10000 # print every 1000 voxels
                if  i % step == 0:
                    crlf = "\r" if total == len(neighbour_idx_list) else "\n"
                    pt = round(float(i)/len(neighbour_idx_list)*100,2)
                    dt = time.time()-t0
                    remaining = (100-pt)/max(0.01,pt)*dt
                    sys.stderr.write(
                        f"job # {thread_id}, processed{i}/{len(neighbour_idx_list)} voxels"
                        f"({pt:0.2f}%, {remaining} seconds remaining){crlf}"
                    )        
        return np.asarray(voxel_results)    

    def genPatches(self,patternimg,use_parallel:bool = True):
        print("generating searchlight patches")
        mask_img_data  = self.mask_img.get_fdata()
        process_coords = np.where(mask_img_data!=0)
        process_coords = np.asarray(
            nilearn.image.coord_transform(
                process_coords[0],
                process_coords[1],
                process_coords[2],
                self.mask_img.affine
                )
            ).T
        def get_patch_data(coords,patternimg):
            X,A = _apply_mask_and_get_affinity(
            seeds  = coords,
            niimg  = patternimg,
            radius = self.radius,
            allow_overlap = True,
            mask_img      = self.mask_img)
            return X,A

        if use_parallel: 
            njobs = cpu_count() - 2            
            split_idx = np.array_split(np.arange(len(process_coords)), njobs)
            pc_chunks = [process_coords[idx] for idx in split_idx]           
            XA_list = Parallel(n_jobs = njobs)(delayed(get_patch_data)(coords,patternimg) for coords in pc_chunks)
            X = XA_list[0][0] # X is the same for each chunk
            A = vstack([l[1] for l in XA_list])
        else:
            X,A = get_patch_data(process_coords,patternimg)
        A = A.tocsr()
        print("finished generating searchlight patches")        
        return X, A 