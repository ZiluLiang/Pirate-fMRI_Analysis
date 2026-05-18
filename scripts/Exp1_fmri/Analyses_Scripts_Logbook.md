This logbook provides a documentation of the analysis scripts associated with the main figures. Details of how the data is preprocessed and prepared as well as how the analyses are conducted can be found in the [Multivariate Analysis Pipeline](/scripts/Exp1_fmri/MultivariateAnalysisPipeline.md).


# Figure 1: Behaviour
## Participant Classification
To compute the LLR during training and test stimuli trials:

1) We first discretised the arena into grids and computed the likelihood of a response falling within each grid following the ground truth or the null model. This is executed in the [``LLR_distribution.Rmd`` file](/scripts/LLR%20Distribution.Rmd).

2) Then, LLR was computed to describe whether the probability that a participant's response followed the ground-truth model ($P_binormal$) exceeded the probability that they were responding randomly ($P_uniform$).  Participants with $LLR_{test}>0$ would be classified as generalizers.  
   
3) We further excluded 4 participants who were learners or generalizers in the last block of pretraining but behaved like non-learners or non-generalizers in the scanner ($LLR_{train}>0$ in the last block of pretraining, $LLR_{train}<>>0$ in scanner; or $LLR_{test}>0$ in pretraining,  $LLR_{test}<0$, see Figure S3-2).  

Step 2-3 are executed in the [``Participant Classification.Rmd``](/scripts/Exp1_fmri/ParticipantClassification.Rmd)  

## Plotting behavioural performance
After classification, behavioural performance were plotted in [``Behavioural Performance.Rmd``](/scripts/Exp1_fmri/BehaviouralPerformance.Rmd)


# Figure 2: Dimensionality analysis
We estimated the functional dimensionality of individual rule representations using a previously established cross-validated SVD procedure, which is implemented in [`computeCVSVDdim_withinaxis.py`](/scripts/Exp1_fmri/multivariate/computeCVSVDdim_withinaxis.py).

LME models examining the relationship between projections and axis locations are conducted in R by [``svdproj.Rmd``](/scripts/Exp1_fmri/svdproj.Rmd)

The decoding analysis included in the partial correlation results was implemented in [`decoding_trainingstim.py`](/scripts/Exp1_fmri/multivariate/decoding_trainingstim.py).

# Figure 3: Parallel and orthogonal representation
We conducted a cross-validated RSA analysis to test the orthogonal vs parallel representation in ROIs. Details can be found in the analysis logbook. This is implemented in [`PTAregRSAtwofold.py`](/scripts/Exp1_fmri/multivariate/PTAregRSAtwofold.py)

For visualisation purposes, we classified participants into different alignment group in [`computeAxisPS_navi.py`](/scripts/Exp1_fmri/multivariate/computeAxisPS_navi.py). Then, we ran MDS on the group average RDM to get the low-dimensional visualisation of neural representation in [`dimreduc_MDS_parallelaxisgroups.py`](/scripts/Exp1_fmri/multivariate/dimreduc_MDS_parallelaxisgroups.py)

# Figure 4: Illustrative model
The Illustrative model in Figure 4 was implemented in Jupyter Notebook [`illustrativeparallelmodel`](/scripts/Exp1_fmri/illustrativeparallelmodel.ipynb).

# Figure 5: Vector addition composition
The retrieval pattern analysis used to test the vector addition hypothesis was implemented in [`compute_train2testRetrievalPatternCV.py`](/scripts/Exp1_fmri/multivariate/compute_train2testRetrievalPatternCV.py)

# Figure 6: Spatial representation
The spatial representation RSA analysis was conducted in both ROI ([`rsa_withintask.py`](/scripts/Exp1_fmri/multivariate/rsa_withintask.py)) and in searchlight (see [`Searchlight-HighLowD.py`](/scripts/Exp1_fmri/Searchlight-HighLowD.py) for participant-level searchlight and [`secondlevelsearchlight.m`](/scripts/Exp1_fmri/secondlevelsearchlight.m) for whole-brain statistics at group level in SPM12).

The cross-task RSA was conducted in ROI ([`rsa_crosstask.py`](/scripts/Exp1_fmri/multivariate/rsa_crosstask.py))

The cross-task decoding was conducted in ROI [`decoding_navi2loc.py`](/scripts/Exp1_fmri/multivariate/decoding_navi2loc.py)

# Control analyses
## In M1
We repeated the dimensionality analyses, vector addition analyses, and spatial representation analyses in M1 in the same scripts specified above.

## Gaze Decoding
Gaze decoding was performed using the DeepMReye toolbox. All scripts are housed in the [deepmreye folder](/scripts/Exp1_fmri/deepmreye/) Our analysis pipeline consists of the following steps:
1. file preparation ([dm_preproc_movefile.py](/scripts/Exp1_fmri/deepmreye/dm_preproc_movefile.py))
2. extracting eye voxels ([dm_preproc_eyemask.py](/scripts/Exp1_fmri/deepmreye/dm_preproc_eyemask.py))
3. preprocessing voxel data for decoding ([dm_preproc_preprocess.py](/scripts/Exp1_fmri/deepmreye/dm_preproc_preprocess.py))
4.obtaining the pretrained models available for download from OSF, applying it to decode gaze from the localiser task data to figure out which model is the best performing one to use for the main treasure hunt task ([dm_decoder_pretrainedmodels.py](/scripts/Exp1_fmri/deepmreye/dm_decoder_pretrainedmodels.py))
5. applying the best performing model to the treasure hunt task ([dm_decoder_m7treasurehunt.py](/scripts/Exp1_fmri/deepmreye/dm_decoder_m7treasurehunt.py))
6. performing shuffling of the treasure hunt task data to do permutation test ([dm_decoder_m7treasurehunt_shuffle.py](/scripts/Exp1_fmri/deepmreye/dm_decoder_m7treasurehunt_shuffle.py))
7. there are two jupyter notebooks that plot the results of pre-trained models selection based on the localiser task [DeepMReye_decodingconsistency](/scripts/Exp1_fmri/deepmreye/DeepMReye_decodingconsistency.ipynb) and the results of applying the selected model to the treausre hunt task [DeepMReye_treasurehuntgaze](/scripts/Exp1_fmri/deepmreye/DeepMReye_treasurehuntgaze.ipynb)
