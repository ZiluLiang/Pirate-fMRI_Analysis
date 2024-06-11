# pirate-fMRI_Analysis
This repo contains codes used for the data analysis of the pirate fMRI project

## Task Design
This task aims to investigate the neural mechanism underlying compositional generalization. This is a follow-up of work by [Dekker 2022, PNAS](https://www.pnas.org/doi/10.1073/pnas.2205582119). Participants are required to navigate the pirate on the screen to find treasure. Treasure location is indicated by the stimulus in each trial. All stimuli have two features (color-shape in experiment 1, shape-shape in experiment 2). Each feature dimension maps onto an axis in the cartesian coordinate system (x/y). This yields a 5-by-5 map. Participants are trained on the central '+' locations on the map and are tested on the other locations.

As of now, the project have two experiments:  
- **Experiment 1**  
A faithful replication of Dekker 2022, PNAS. Stimuli are generated from the factorial combinations of colours and shapes. Participants only learned one map.

- **Experiment 2**  
To further tease apart the influence of learning-induced factorisation. Stimuli are generated from factorial combinations of two sets of shapes. In different pilots, participants either learned one map or two maps.

For details of the task design see [Task Design]()


## Pipelines
### **Experiment 1**  
Overview and diary of analysis @ [AnalysisPlanNoteBook](/docs/AnalysisPlanNoteBook.md) 
1. [fmri data preprocessing pipeline](/src/preprocessing/PreprocessingPipeline.md)
2. behavioral data analysis pipeline
3. fmri data analysis pipelines: [univariate](/src/univariate/UnivariateAnalysisPipeline.md), [multivariate](/src/multivariate/MultivariateAnalysisPipeline.md) 

### **Experiment 2**  
1. behavioral data anlaysis pipeline


## Prerequisites
### Platform and Toolbox:
**MATLAB R2023a**:
- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [jsonlab](https://uk.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files)
- [marsbar](https://marsbar-toolbox.github.io/)  

[**MRIcroGL**](https://www.nitrc.org/projects/mricrogl) 
python needs to be installed for customised MRIcroGL scripts to run.  

[**Python environment** (and packages)](/piratefmri.yml)

### Installing python environment from yaml file
use `conda env create -f piratefmri.yml` to install the packages used in this repo.  
**Troubleshot:**
1. matlabengine version not supported  
The yaml file specified a version of matlabengine that is supported with MATLAB R2023a and python 3.9.16, if you have other version of MATLAB and python installed, you should change it to the matching matlabengine version. To check that [visit MATLAB's official help page](https://nl.mathworks.com/support/requirements/python-compatibility.html)

2. installing zpyhelper  
zpyhelper is an accompanying package for running MVPA analysis in python that is still under development. It is not currently released yet. To install, you need to go to [the GitHub repo for zpyhelper](https://github.com/ZiluLiang/zpyhelper) and follow the instruction.

### File structure
see [File Structure](/docs/FILESTRUCTURE.md) for naming conventions and data structure