## Activity pattern
To run multivariate analysis, we need to get the activity pattern matrix. To achive that, we adopted the ['LS-A' approach](https://doi.org/10.1016/j.neuroimage.2011.08.076).
### 1. the LS-A approach
For each participant, we build first level GLMs with 25 regressors (one for each stimuli) in each session (run), this yields a totoal of 25*4 = 100 condition regressors. Response stage was modeled with a boxcar function with a duration corresponding to participant's actual navigation time in the task. Realignment parameters and their first derivatives are included as nuisance regressors (see [Univariate Analaysis Pipeline](/scripts/univariate/UnivariateAnalysisPipeline.md)).  
Then we build the following contrasts:  
(1) a contrast for each of the stimuli, this yields 25 contrasts  
We extracted the 25 stimuli contrast to constructs a $N_{stimuli}\times N_{voxels}$ activity pattern matrix for visualization.

(2) a contrast for each of the stimuli in odd/even runs, this yields 25*2 = 50 contrasts  
We extracted the 50 stimuli contrasts separating odd and even runs to construct two $N_{stimuli}\times N_{voxels}$ activity pattern matrices for calculating reliability map and running RSA.

### 2. activity pattern 4D nii files
Then three 4D nii files are created by combining the following three groups of images together:  
(1) a `stimuli_4r.nii` file created from 100 beta_\*\*\*\*.nii files, each representing the beta estimates of one stimulus in one run. In the combined 4D nii file, the volumes are ordered according to:  
stim00run1, stim01run1, ..., stim00run2, stim01run2, ..., stim00run4, ..., stim24run4  
(2) a `stimuli_oe.nii` file created from 50 con_\*\*\*\*.nii files, each representing the contrast estimates of one stimulus in odd/even run. In the combined 4D nii file, the volumes are ordered according to:  
stim00odd, stim01odd, ..., stim24odd, stim00even, stim01even, ...  stim24even  
(3) a `stimuli_mu.nii` file created from 25 con_\*\*\*\*.nii files, each representing the contrast estimates of one stimulus across all runs. In the combined 4D nii file, the volumes are ordered according to:  
stim00, stim01, ..., stim24

### 3. Reliability Map calculation
Using a [split-half approach](https://doi.org/10.1016/j.neuroimage.2019.116350), we calculate the reliability map for each participant.
Let $M1$ and $M2$ be the $N_{stimuli}\times N_{voxels}$ activity pattern matrices for the odd and even runs respectively. Then $M1_{i}$ and $M2_{i}$ (column $i$ in $M1$ and $M2$) defines the response profile of voxel $i$ in odd and even runs. Reliability is defined as the stability of the response profile, i.e., the reliabity of a given voxel $i$ is computed by  
$$R_{i} = pearsonr(M1_{i},M2_{i})$$ 
A nifti file of the reliability map is saved in the first level directory of the participants. Then a threshold of 0 is applied to binarize the map into a reliability mask which specifies the reliable voxels.
This is implemented in the [`create_reliability_mask` script](/scripts/multivariate/create_reliability_mask.py)

## Brain parcellation based RSA
### obtaining ROI mask from parcellations 
### running analysis 
## Whole brain searchlight RSA