# pirate-fMRI_Analysis
This repo contains codes used for the data analysis of the pirate fMRI project
## Task Design
This task aims to investigate the neural mechanism underlying compositional generalization. For a detailed description see [Dekker 2022, PNAS](https://www.pnas.org/doi/10.1073/pnas.2205582119). Participants are required to navigate the pirate on the screen to find treasure. Treasure location is indicated by the stimulus in each trial. All stimuli are two-dimensional, i.e. colored shape. Each feature dimension maps onto an axis in the cartesian coordinate system (x/y). Participants are trained on the 9 '+' locations on the map and are tested on the other 16 locations.

## Pipelines
1. [fmri data preprocessing pipeline](scripts/preprocessing/PreprocessingPipeline.md)
2. behavioral data analysis pipeline
3. fmri data analysis pipelines: [univariate](/scripts/univariate/UnivariateAnalysisPipeline.md), [multivariate](/scripts/multivariate/MultivariateAnalysisPipeline.md)

## Prerequisites
### Platform and Toolbox:
[SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) in MATLAB2020b  
[MRIcroGL](https://www.nitrc.org/projects/mricrogl) python needs to be installed for customised MRIcroGL scripts to run.
### File structure
see [File Structure](FILESTRUCTURE.md) for naming conventions and data structure