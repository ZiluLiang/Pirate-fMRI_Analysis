## Model Overview
For our univariate analysis, first-level usually model the stimuli stage and the response stage. Depending on the conditions, they maybe further split into different regressors. For the stimuli presentation stage, a boxcar function is used, with duration corresponding to the length of stimuli presentation. For the response stage in navigation task, a boxcar function with participant's actual navigation time as duration was used. For the response stage in the localizer task, a stick function was used (i.e., duration = 0).

## Detailed Model Designs
Different univariate analysis models different conditions. The following classes of univariate analyses have been performed.
### I. Sanity check models
1. sanity check for navigation task  
   - **First-level**: each participants 4D fmri series are modeled with two regressors: stimuli and response. box car functions are used with duration corresponding to actual stimuli presentation time and actual response time (time participants spent moving the pirate)
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

### II. Repetition Suppression models
1. Repetition suppression for navigation task  
   Repetition suppression models for the navigation task modeled three events:  
   (1) the stimuli event for trials not preceded by a response trial   
   (2) the motor event in response trial  
   (3) the stimuli event for trials preceded by a response trial    

   In the first regressor, distance between current trial stimuli and previous trial stimuli is included as a parametric modulator for the stimuli event.  Several variants of this model with different various distance metric is constructed:  
   (1) with groundtruth euclidean distance  
   (2) with euclidean distance computed from participants' response map.

2. Repetition suppression model for the localizer task
   This is similar to the one above for navigation task, except that button presses are not modeled with boxcar function, a stick function is used instead. 

### III. Models for extracting beta series
To extract activity related to stimuli, we adopted the LSA approach. We build first level GLM in which each stimuli is modeled with one regressor in each session. This yields 25 stimuli regressors for each session in the navigation task (100 beta series in total) and 9 stimuli regressors for the localizer task (one session, only 9 training locations are showed in the localizer task). Similar to other GLMs, the response is modeled with a separate regressor, either using boxcar function (navigation task) or stick function (localizer task). For more details of the RSA analysis please refer to [MultivariateAnalysisPipeline](/src/multivariate/MultivariateAnalysisPipeline.md)

## SPM settings
All univariate analysis are conducted with canonical HRF without temporal and dispersion derivatives. We use spm's intracranial volume mask as explicit mask (mask_ICV.nii in the `spm/tpm` directory) combined with a lowered implicit masking threshold of 0.2. 

## Scripts Overview
Three main scripts runs univariate analysis: 
- [`sanity_check.m`](/scripts/Exp1_fmri/sanity_check.m) runs sanity check on the data, while the
- [`univariate_analysis.m`](/scripts/Exp1_fmri/univariate_analysis.m) performs the repetition suppression analysis.
- [`prepare_MVPA`](/scripts/Exp1_fmri/prepare_MVPA.m) builds the models for extracting beta series.

The functions in [`univariate`](/src/univariate/) subfolder specify the batch jobs for running glm analysis in SPM:  
- [`glm_firstlevel`](/src/univariate/glm_firstlevel.m) specifies the first level glm models. 
- [`glm_concatenate`](/src/univariate/glm_concatenate.m) concatenates separate runs into one long run before estimation. 
- [`glm_grouplevel`](/src/univariate/glm_grouplevel.m) specifies the group level glm models. 
- [`glm_estimate`](/src/univariate/glm_estimate.m) estimates specified glm models.  
- [`glm_contrast`](/src/univariate/glm_contrast.m) specifies contrasts for estimated glm models. 
- [`glm_results`](/src/univariate/glm_results.m) shows results reports on all the contrasts in corresponding SPM.mat file, can be first/second level.   

Models are configured in [`glm_configure`](/src/univariate/glm_configure.m) and configurations are identified by the name of glms. Three main scripts all calls [`glm_runner`](/src/univariate/glm_runner.m) to run different steps in glm analysis. It first reads in the configurations, prepare relevant directories and then calls the above batch-job scripts to set up batch job for analysis. 