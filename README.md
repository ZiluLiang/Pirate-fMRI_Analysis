# pirate-fMRI_Analysis
This repo contains codes used for the data analysis of the pirate fMRI project

## Task Design
This task aims to investigate the neural mechanism underlying compositional generalization. This is a follow-up of work by [Dekker 2022, PNAS](https://www.pnas.org/doi/10.1073/pnas.2205582119). Participants are required to navigate the pirate on the screen to find treasure. Treasure location is indicated by the stimulus in each trial. All stimuli have two features (color-shape). Each feature dimension maps onto an axis in the cartesian coordinate system (x/y). This yields a 5-by-5 map. Participants are trained on the central '+' locations on the map and are tested on the other locations.

## Pipelines
To find out the correspondence between the scripts and the analyses reported in the paper, see [Analyses-Scripts-Logbook](/scripts/Exp1_fmri/Analyses_Scripts_Logbook.md).

Detailed pipelines for the analyses are documented here:
1. [fmri data preprocessing pipeline](/src/preprocessing/PreprocessingPipeline.md)
2. [fmri data multivariate analyses pipelines](/scripts/Exp1_fmri/MultivariateAnalysisPipeline.md) 


## Prerequisites
### Platform and Toolbox:
**MATLAB R2023a**:
- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [jsonlab](https://uk.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files)
- [marsbar](https://marsbar-toolbox.github.io/)  

[**MRIcroGL**](https://www.nitrc.org/projects/mricrogl) 
python needs to be installed for customised MRIcroGL scripts to run.  

**Python environment** can be directly installed from yaml file [here](/piratefmri.yml), then install [zpyhelper](https://github.com/ZiluLiang/zpyhelper) into this environment. Steps: 
- 1. use `conda env create -f piratefmri.yaml` to install the packages used in this repo.  
- 2. then go to [the GitHub repo for zpyhelper](https://github.com/ZiluLiang/zpyhelper), download the code to local directory, and install in the 'piratefmri' environment.
        ```
        cd zpyhelper_dir
        conda activate piratefmri
        pip install -e.
        ```

### **Troubleshot:**
1. matlabengine version not supported  
The yaml file specified a version of matlabengine that is supported with MATLAB R2023a and python 3.9.16, if you have other version of MATLAB and python installed, you should change it to the matching matlabengine version. To check that [visit MATLAB's official help page](https://nl.mathworks.com/support/requirements/python-compatibility.html)

2. installing zpyhelper  
zpyhelper is an accompanying package for running MVPA analysis in python that is still under development. It is not currently released yet. To install, you need to go to [the GitHub repo for zpyhelper](https://github.com/ZiluLiang/zpyhelper) and follow the instruction.

### File structure
see [File Structure](/docs/FILESTRUCTURE.md) for naming conventions and data structure