# File Structure
Files are organized in the following structure:
```
├── data
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
├── plots
└── scripts
    ├── preprocessing
    ├── qualitycheck
    ├── utils
    ├── get_pirate_defaults.m
    ├── prepare_image.m
    ├── preprocess_image.m
    └── xx analysis.m
```
## scripts 
Scripts for different analysis (including preprocessing) are placed in `scripts` directory while its dependencies are placed in subfolders of the `scripts` directory.  

## data
### data naming convention
data are named in a BIDS-like format. Future work will try to make it compatible with BIDS.  
for the raw fmri data in `the fmri/data/renamed` folder, file names are constructed from several key-value pairs:
sub-xxx_modality-xxx_run-xx.nii
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

### data structure
Data are all placed in the data folder, sorted according to data type and analysis type.
`data` .  
1. `data/renamer.json`: json files specifying how to annoymize and rename behavioral and fmri data. 
  
`data/fmri` contains fmri data, raw or after different processing/analylizing pipeline.  
1. `data/fmri/rename`: renamed, organized raw fmri data (unpreprocessed) 
2. `data/fmri/preprocess`: files created after different preprocessing steps
3. `data/fmri/unsmoothed`: preprocessed unsmoothed fmri data
4. `data/fmri/smoothed`: preprocessed smoothed fmri data
5. `data/fmri/preprocess-automated`: legacy data. fmri data preprocessed using pipeline that does not include reorientation in the beginning. This pipeline was deprecated and replaced by the pipeline that include auto reorientation in the beginning. For more details of the final pipeline, see [`PreprocessPipeline.md`](scripts/preprocessing/PreprocessingPipeline.md).
 

 
 

