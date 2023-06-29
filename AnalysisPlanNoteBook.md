This is the analysis plan/log notebook for the pirate project. It keeps tracks of hypotheses we intended to test and different analysis plans we designed to test these hypotheses.

# Hypotheses
## Hypothesis 1: If Hippocampus(HPC), prefrontal cortex(PFC), and maybe post-parietal cortex (PPC) represent the map, then the representation similarity of the stimuli will be predicted by their distance on the treasure map - RSA analysis
### Literature
tbc
### Analysis
searchlight and parcellation-based RSA analysis

#### 1. preparing the activity pattern matrix
There are several options we can tune in our pipeline when preparing the activity pattern matrix. Different pipelines are compared to see the robustness of findings and whether the results yield by certain pipeline is biased. Sanity check models are deigned to see if pipelines yield sensible results.
##### (1) building first level GLM to extract activity pattern matrix
The LSA approach: We can use either all four runs, or we can concatenate (using spm_concatenate) to do it.
The default way SPM deals with it is to do it in four runs, this models each run separately, we will have 25*4=100 regressors in total. The upside of this is that we can do cross-validated RSA analysis. The downside is that this may suffer from low estimation power, i.e., estimating too many parameters in too few samples yields unstable parameter estimates. An alternative is to use spm_concatenate to concatenate all four runs together, then we will only have 25 regressors, each will be estimated with 3 times more samples compared to the previous approach. The downside is that the last scan from a previous run and the first scan of a new run will be treated as consecutive scans. Nonetheless both approaches can be used to compare their consistency. 
##### (2) voxel selection
select voxel with spearman rank correlation>0 ([details of selection procedure](/scripts/multivariate/MultivariateAnalysisPipeline.md#3-reliability-map-calculation))
plot the distribution of correlation in each participants. Currently ~50% of voxels are retained after selection in unsmoothed data. But in the smoothed data, this number is more variable, ranging from 30%-70%. This could be due to the unreliable parameter estimate or just that smoothing introduces more spatial correlation and push things to the extreme ends. So for now, unsmoothed data is used/

Another issue is, with voxel selection, almost all the effect becomes stronger (in ROI-based analysis and whole-brain searchlight), we need to check if voxel selection biased our results and lead to false positive. To check if voxel selection biased our analysis, two type of permutation are conducted:
(1) permute a random voxel selection mask with the same number of voxels as the main mask
(2) permute a random rdm and compare the result of voxel selection
In addition, control regions are setup to check if the voxel selection create positive bias or spurious correlation

#### 2. where to conduct RSA analysis
RSA is conducted both within [ROI](/scripts/multivariate/MultivariateAnalysisPipeline.md#1-brain-parcellation-based-rsa-obtaining-roi-masks-from-aal-parcellation) and at [whole-brain level](/scripts/multivariate/MultivariateAnalysisPipeline.md#2-whole-brain-searchlight-rsa-obtaining-spherical-searchlight-regions) to check the robustness of the result and compare different pipelines.

#### 3. computing neural RDMs
Neural RDM is calculated based on the activity pattern we obtained in [step 1](/AnalysisPlanNoteBook.md#1-preparing-the-activity-pattern-matrix). However some 'preprocessing' on this neural RDM is necessary to remove noise or control for confounds.  
Several decisions to make when computing neural RDM:  
(1) whether or not to do centering  
(2) which distance metric to use  
Three common metrics of similarity metric:  
- Correlation  
- Euclidean distance
- Mahalanobis distance - this requires estimation of a covariance matrix. should do more reading and decide if it is a more suitable alternative. list of literatures:
   - Bobadilla-Suarez et al. [Measures of Neural Similarity](https://doi.org/10.1007/s42113-019-00068-5)[<sup>2</sup>](/AnalysisPlanNoteBook.md#references) and [open codes](https://osf.io/5a6bd/?view_only=)
   - Kriegeskorte's [blog post] on the above paper (https://nikokriegeskorte.org/category/representational-similarity-analysis/)
   - [Walther et al.,](https://doi.org/10.1016/j.neuroimage.2015.12.012)[<sup>1</sup>](/AnalysisPlanNoteBook.md#references) showed that cross-validated Mahalanobis distance is more suitable.
   - [rsatoolbox's documentation and implementation of Mahalanobis distance](https://rsatoolbox.readthedocs.io/en/latest/distances.html)
   - Maybe [this unpublished tutorial by Walther et al](https://www.mrc-cbu.cam.ac.uk/wp-content/uploads/www/sites/3/2014/10/Walther_etAl_representationalfMRIanalysis_unpublishedDraft.pdf)?
   - [An overview by Diedrichsen, J., & Kriegeskorte](https://doi.org/10.1371/journal.pcbi.1005508)[<sup>3</sup>](/AnalysisPlanNoteBook.md#references) that seems helpful

#### 6. computing model RDMs
(1) sanity check model RDMs  
(2) feature-based model RDMs  
(3) map-based model RDMs  
## Hypothesis 2: color or shape will be represented by two orthogonal neural axis - Decoding analysis
### Literature
tbc
### Analysis
use neural activity to predict locations
split train/evaluation set by: actual train/test, random row/col, cross-task generalization (localizer to navigation, navigation to localizer)


# References
1. [Walther, A., Nili, H., Ejaz, N., Alink, A., Kriegeskorte, N., & Diedrichsen, J. (2016). Reliability of dissimilarity measures for multi-voxel pattern analysis. Neuroimage, 137, 188-200.](https://doi.org/10.1016/j.neuroimage.2015.12.012)  
2. [Bobadilla-Suarez, S., Ahlheim, C., Mehrotra, A., Panos, A., & Love, B. C. (2020). Measures of neural similarity. Computational brain & behavior, 3, 369-383.](https://doi.org/10.1007/s42113-019-00068-5)  
3. [Diedrichsen, J., & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis. PLoS computational biology, 13(4), e1005508.](https://doi.org/10.1371/journal.pcbi.1005508)