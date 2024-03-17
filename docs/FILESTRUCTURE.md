# File Structure
Files are organized in the following structure:
```
├── data
    ├── Exp1_fmri
        ├── behavior
        └── fmri
            ├── renamed
                ├── sub001
                ├── ...
                ├── sub00n
            ├── preprocess
                ├── sub001
                ├── ...
                ├── sub00n
            ├── unsmoothed
                ├── sub001
                ├── ...
                ├── sub00n
            └── xx analysis
    └── Exp2_pilots
├── scripts
    ├── LLR_check
    ├── Exp1_fmri
    └── Exp2_pilots
├── results
    ├── LLR_check
    ├── Exp1_fmri
    └── Exp2_pilots
└── src
    ├── preprocessing
    ├── qualitycheck
    ├── utils
    ├── get_pirate_defaults.m
    ├── prepare_image.m
    ├── preprocess_image.m
    └── xx analysis.m
```
## scripts and results 
Scripts for different analysis are placed in `scripts` directory while its results (such as plots) are placed in the `results` directory.  
This can be analysis of a single experiment or across experiment.

## src
Reusable modules, i.e. dependencies.

## data
Data is saved in different experiment folder. For instance, `Exp1_fmri` is the folder hosting all the data from experiment 1

### data naming convention
data are named in a BIDS-like format. Future work will try to make it compatible with BIDS.  

For the raw fmri data in the `fmri/data/renamed` folder. nii files are usually accompanied by json files that stores meta information. File names are constructed from several key-value pairs:
``sub-xxx_modality-xxx_run-xx.nii``  
``sub-xxx_modality-xxx_run-xx.json``  
for instance:  
```
└── sub001
    ├── sub-001_anat-T1w.nii
    ├── sub-001_anat-T1w.json
    ├── sub-001_fmap-magnitude1.nii
    ├── sub-001_fmap-magnitude1.json
    ├── sub-001_fmap-magnitude2.nii
    ├── sub-001_fmap-magnitude2.json
    ├── sub-001_fmap-phasediff.nii
    ├── sub-001_fmap-phasediff.json
    ├── sub-001_task-piratenavigation_run-1.nii
    ├── sub-001_task-piratenavigation_run-1.json
    ├── ...
    ├── sub-001_task-localizer_run-1.json
    ├── sub-001_task-localizer_run-1.json
    └── sub-001_fmap-magnitude2.json
```
During preprocessing, each step will add its own prefix to the nii file (prefix can be found at [get_pirate_defaults](/./scripts/Exp1_fmri/get_pirate_defaults.m)), no json file is created.

For the behavioral data, the naming follows the same convention, without the json files.


### data structure
Data are all placed in the data folder, sorted according to data type and analysis type.
`Exp1_fmri` is the folder hosting all the data from experiment 1
1. `Exp1_fmri/renamer.json`: json files specifying how to annoymize and rename behavioral and fmri data. 
  
`datExp1_fmria/fmri` contains fmri data, raw or after different processing/analylizing pipeline.  
1. `Exp1_fmri/fmri/rename`: renamed, organized raw fmri data (unpreprocessed) 
2. `Exp1_fmri/fmri/preprocess`: files created after different preprocessing steps
3. `Exp1_fmri/fmri/unsmoothed`: preprocessed unsmoothed fmri data
4. `Exp1_fmri/fmri/smoothed`: preprocessed smoothed fmri data
5. `Exp1_fmri/fmri/preprocess-automated`: legacy data. fmri data preprocessed using pipeline that does not include reorientation in the beginning. This pipeline was deprecated and replaced by the pipeline that include auto reorientation in the beginning. For more details of the final pipeline, see [`PreprocessPipeline.md`](/./src/preprocessing/PreprocessingPipeline.md).
 

 
 

