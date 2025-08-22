This logbook provides a documentation of the analysis script associated with the main figure. Details of how the analysis are conducted can be found in the [Multivariate Analysis Pipeline](/scripts/Exp1_fmri/MultivariateAnalysisPipeline.md).

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
