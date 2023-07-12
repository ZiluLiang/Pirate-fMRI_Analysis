## Activity pattern
To run multivariate analysis, we need to get the activity pattern matrix. To achieve that, we adopted the ['LS-A' approach](https://doi.org/10.1016/j.neuroimage.2011.08.076).
### 1. the LS-A approach
For each participant, we build first level GLMs with 25 regressors (one for each stimuli) in each session (run), this yields a total of 25*4 = 100 condition regressors. Response stage was modeled with a boxcar function with a duration corresponding to participant's actual navigation time in the task. Realignment parameters and their first derivatives are included as nuisance regressors (see [Univariate Analysis Pipeline](/scripts/univariate/UnivariateAnalysisPipeline.md)).  
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
Let $M1$ and $M2$ be the $N_{stimuli}\times N_{voxels}$ activity pattern matrices for the odd and even runs respectively. Then $M1_{i}$ and $M2_{i}$ (column $i$ in $M1$ and $M2$) defines the response profile of voxel $i$ in odd and even runs. Reliability is defined as the stability of the response profile, i.e., the reliability of a given voxel $i$ is computed by  
$$R_{i} = Spearman's r(M1_{i},M2_{i})$$ 
A nifti file of the reliability map is saved in the first level directory of the participants. Then a threshold of 0 is applied to binarize the map into a reliability mask which specifies the reliable voxels.
This is implemented in the [`create_reliability_mask` script](/scripts/multivariate/create_reliability_mask.py)

## Neural Vector Analysis
Let $𝑓_𝑗$ be the coordinate of stimuli $j$ in neural representation space $ℝ^𝑚$ ($m$ being the number of voxels). For any two stimuli $𝑗,𝑘$, we define the neural vector (coding direction) from $𝑗$ to $𝑘$ as:  
$$
𝑣_{𝑗𝑘}=𝑓_𝑗− 𝑓_𝑘
$$
For any given pair of neural vectors, we can compute its cosine similarity as an indicator of how parallel these two neural vectors are
$$
cos⁡(𝑣_1, 𝑣_2)=  \frac{𝑣_1 \times 𝑣_2}{\left\Vert 𝑣_1 \right\Vert \times \left\Vert 𝑣_2 \right\Vert}
$$
If we replace $𝑓_𝑗$   and $𝑓_𝑘$ with the feature vector from the three models of representation (row vector from the feature matrices), we can compute the theoretical cosine similarity of any given pair of coding directions, and compare that with our data.
![feature matrices for different models of representation](<img src="/plot/featurematrix_by_representationmodels.png"/>)  


## RSA
### Regions
#### 1. Brain parcellation based RSA: obtaining ROI masks from AAL parcellation
AAL3v1 parcellation is used to generate anatomical masks for ROIs. The procedure was carried out in marsbar. Each anatomical masks is a binarize and resampled to match the resolution of the participants' functional images.  
Details on what parcellation is included in each anatomical mask are in [`anatomical_masks.json`](/scripts/anatomical_masks.json). This is read in by [`anatomical_masks.m`](/scripts/anatomical_masks.m) to generate the mask file in nii format. 

#### 2. Whole brain searchlight RSA: obtaining spherical searchlight regions
For each voxel, a spherical ROI is defined with a radius of 10mm (4 times the voxel size). An additional constraint is added: each searchlight sphere should include at least 50 usable voxels. 

### Neural RDM
activity pattern within each mask/sphere is extracted to compute the neural RDM. If voxel selection is performed, then the spherical region will only include the selected voxels to compute neural RDM. 
(1) Centering
Before computing neural RDM, centering is performed for each voxel separately.
(2) Distance metrics
- Euclidean
- Correlation: this is the one we are currently using
- Mahalanobis distance 
 
### Quantifying similarity between neural RDM and model RDM
Only the lower triangular part of the model RDMs and neural RDMs (excluding diagonals) are extracted for the analysis
(1) regression
regression uses one or a set of model RDMs to predict neural RDM. Dependent variable and predictors are standardized (separately) before entering the regression analysis.
(2) correlation
Spearman's rank correlation is computed between the neural RDM and one model RDM. the correlation coefficient is Fisher z-transformed before entering second level analysis

### Statistical tests
#### Parametric tests
For ROI-based analysis one sample t-test is performed on the metric
For whole-brain searchlight analysis, for a given predictor, a metric map is generated for each participant. This is then entered into SPM's second level analysis to perform statistic test (one-sample t-test) and multiple correction.
#### Non-parametric tests with permutation test
TBC
non parametric test do not assume distribution of the metric, and maybe a more exact test for the effect. We can shuffle the activity pattern matrix and generate null distribution of the metric and compare our estimated value to this null distribution.