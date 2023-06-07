## Model Designs
The following classes of univariate analyses have been performed
### Sanity check models
1. sanity check for navigation task  
   - **First-level**: each participants 4D fmri series are modeled with two regressors: stimuli and response. box car functions are used with duration corresponding to acutual stimuli presentation time and actual response time (time participants spent moving the pirate)
   two T-contrasts are built:  
    (1) to test for visual effect: a weight vector of `[1,1]` is used  
    (2) to test for motor effect: a weight vector of `[0,1]` is used  
    - **Second-level**: each first-level contrast from all participants are selected, group effect is tested using one-sample t-test.
2. sanity check for localizer task  
   - **First-level**: each participants 4D fmri series are modeled with two regressors: stimuli and response. Stick functions are used for both regressors.
   two T-contrasts are built:    
    (1) to test for visual effect: a weight vector of `[1,0]` is used   
    (2) to test for motor effect: a weight vector of `[0,1]` is used   
   - **Second-level**: each first-level contrast from all participants are selected, group effect is tested using one-sample t-test.  

### Repetition Supression models
Repetition supression models for the navigation task modeled three events:
(1) the stimuli event for trials not preceeded by a response trial   
(2) the motor event in response trial  
(3) the stimuli event for trials preceeded by a response trial  
In the first regressor, the euclidean distance between current trial stimuli and previous trial stimuli is included as a parametric modulator for the stimuli event.  
Two variants of this model is constructed, one with groundtruth euclidean distance, the other with euclidean distance computed from participants' response map.

Repetition supression model for the localizer task is similar to the one above for navigation task, except that button presses are not modeled. Only one model with euclidean distance between pirate locations in successive trials are included.

### Models for extracting beta series
To extract activity related to stimuli, we adopted the LSA approach. We build first level GLM in which each stimuli is modeled with one regressor in each session. This yields 25 stimuli regressors for each session in the navigation task (100 beta series in total) and 9 regressors for the localizer task (oen session, only 9 training locations are showed in the localizer task). For more details of the RSA analysis please refer to [MultivariateAnalysisPipeline](scripts/multivariate/MultivariateAnalysisPipeline.md)

## SPM settings
All univariate analysis are conducted with canonical HRF without temporal and dispersion derivatives. We use spm's intracranial volume mask as explicit mask (mask_ICV.nii in the `spm/tpm` directory) combined with a lowered implicit masking threshold of 0.2. 

## Scripts Overview
Two main scripts runs univariate analysis: the [`sanity_check.m`](/scripts/sanity_check.m) main script runs sanity check on the data, while the [`univariate_analysis.m`](/scripts/univariate_analysis.m) main script performs the repetition supression analysis and build the model for extracting beta series.

The following functions specify the batch job for running glm analysis in SPM:  
- [`specify_estimate_glm`](/scripts/univariate/specify_estimate_glm.m) specify and/or runs estimation on the first level glm models. 
- [`specify_estimate_grouplevel`](/scripts/univariate/specify_estimate_grouplevel.m) specify and/or runs estimation on the second level glm models.  
- [`specify_estimate_contrast`](/scripts/univariate/specify_estimate_contrast.m) specify and estimate contrast of first or second level glm models. 
- [`report_results`](/scripts/univariate/report_results.m) shows results reports on all the contrasts in corresponding SPM.mat file, can be first/second level. 


Models are configured in [`get_glm_config`](/scripts/univariate/get_glm_config.m) and configurations are identified by the name of glms. the [`run_glm`](/scripts/univariate/run_glm.m) runs different steps in glm analysis. It first reads in the configurations, prepare relevant diretories and then calls the above batch-job scripts to set up batch job for analysis. 