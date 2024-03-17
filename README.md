# pirate-fMRI_Analysis
This repo contains codes used for the data analysis of the pirate fMRI project

## Task Design
This task aims to investigate the neural mechanism underlying compositional generalization. For a detailed description see [Dekker 2022, PNAS](https://www.pnas.org/doi/10.1073/pnas.2205582119). Participants are required to navigate the pirate on the screen to find treasure. Treasure location is indicated by the stimulus in each trial. All stimuli are two-dimensional, i.e. colored shape. Each feature dimension maps onto an axis in the cartesian coordinate system (x/y). Participants are trained on the 9 '+' locations on the map and are tested on the other 16 locations.

As of now, the project have two experiments:
- Exp1
A faithful replication of Dekker 2022, PNAS. Stimuli are generated from the factorial combinations of colours and shapes. Participants only learned one map.

- Exp2
To further tease apart the influence of learning-induced factorisation. Stimuli are generated from factorial combinations of two sets of shapes. In different pilots, participants either learned one map or two maps.

## Pipelines
For experiment 1:
Overview and diary of analysis @ [AnalysisPlanNoteBook](/docs/AnalysisPlanNoteBook.md) 
1. [fmri data preprocessing pipeline](/src/preprocessing/PreprocessingPipeline.md)
2. behavioral data analysis pipeline
3. fmri data analysis pipelines: [univariate](/src/univariate/UnivariateAnalysisPipeline.md), [multivariate](/src/multivariate/MultivariateAnalysisPipeline.md) 


## Prerequisites
### Platform and Toolbox:
**MATLAB R2023a**:
- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [jsonlab](https://uk.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files)
- [marsbar](https://marsbar-toolbox.github.io/)  

[**MRIcroGL**](https://www.nitrc.org/projects/mricrogl) (python needs to be installed for customised MRIcroGL scripts to run.)
[**Python environment** (and packages)](/piratefmri_condaenv.txt)

### File structure
see [File Structure](/docs/FILESTRUCTURE.md) for naming conventions and data structure