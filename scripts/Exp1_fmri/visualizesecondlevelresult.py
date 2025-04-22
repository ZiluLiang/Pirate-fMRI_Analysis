"""
This script is used to visualize the second-level results of the searchlight RSA analysis.
Note that it cannot be ran directly in the python console, but should be ran in MRIcroGL.
"""
from copy import deepcopy
import os
import gl


import warnings
warnings.simplefilter('ignore', category=FutureWarning)

project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')

navi_dir = os.path.join(fmridata_dir,'unsmoothedLSA','rsa_searchlight','fourruns_mvnn_averageall')
loc_dir = os.path.join(fmridata_dir,'unsmoothedLSA','rsa_searchlight','localizer_mvnn_averageall')

searchlightrsa_dirs = {
    "teststimhighlowd":os.path.join(navi_dir,'regression','compete_featurecartesian_combinexy_teststim','second'),
    "trainingloc":     os.path.join(loc_dir,'correlation','HighLowD','second','gtlocEuclidean','C1C2_G'),
    "traintest":       os.path.join(navi_dir,'regression','compete_featurecartesian_combinexy_withsg','second')
}
Tmapdirs = {"Parietal_teststim2D":    [os.path.join(searchlightrsa_dirs["teststimhighlowd"],'teststimpairs_gtlocEuclidean','C1C2_G','Threshvoxel001cluster30vx_spmT_0001.nii')],	
            "Parietal_trainingloc2D": [os.path.join(searchlightrsa_dirs["trainingloc"],'Threshvoxel0001clusterfdr_spmT_0001.nii')],
            "Parietal_traintestSep":  [os.path.join(searchlightrsa_dirs["traintest"],'stimuligroup','C1C2_G','Threshvoxel001cluster111_spmT_0001.nii')], #or "Threshvoxel0001clusterfdr05_spmT_0001.nii" "Threshvoxelfwe_spmT_0001.nii"
            "Visual_locationrep":  [os.path.join(searchlightrsa_dirs["trainingloc"],'Threshvoxel0001clusterfdr_spmT_0001.nii'),
                                    os.path.join(searchlightrsa_dirs["traintest"],'gtlocEuclidean','C1C2_G','Threshvoxel001cluster194vx_spmT_0001.nii')
                                    ],   
            "ParietalPFC_traintest001": [os.path.join(searchlightrsa_dirs["traintest"],'stimuligroup','C1C2_G','Threshvoxel001cluster111_spmT_0001.nii')]                       
            }
Tmapviews = {"Parietal_teststim2D":    [-22.5,-61,56],#[24, -56,  65],
             "Parietal_trainingloc2D": [-22.5,-61,56],
             "Parietal_traintestSep":  [-22.5,-61,56],
             "Visual_locationrep":     [12,-61,10],
             "ParietalPFC_traintest001":[-5,-61,-2]
             }
Tranges = {"Parietal_teststim2D":[3.31,10], # t=3.31 is the cutoff for p=0.001
           "Parietal_trainingloc2D":[3.31,10], # we chose an image with cutoff at p=.0005, but we use t=3.31 (the cutoff for p=0.001) bc we are showing parietals all together, we keep the same range for easy comparison
           "Parietal_traintestSep":[3.31,10], # t=3.31 is the cutoff for p=0.001
           "Visual_locationrep":[3.55,10],    # t=3.55 is the cutoff for p=0.001
           "ParietalPFC_traintest001":[3.31,10] # t=3.31 is the cutoff for p=0.001
           }
savedir = os.path.join(fmridata_dir,'unsmoothedLSA','rsa_searchlight','Tmaps')

def plot_Tmap(Tmapfps,views,outputfn,Trange=None,colorschemes=None):
    colorschemes = ["4hot"]*len(Tmapfns) if colorschemes is None else colorschemes
    #open background image
    gl.loadimage("spm152")
    gl.minmax(0,10,80)
    gl.colorname(0,"Grayscale")
    
    #open overlays:
    for j, (Tmapfp, cs) in enumerate(zip(Tmapfps,colorschemes)):
        gl.overlayload(Tmapfp)
        if Trange is not None:
            gl.minmax(j+1,Trange[0],Trange[1])
        #set color palette
        gl.colorname(j+1,cs)
    
    #set view coordinates
    gl.orthoviewmm(*views)
    
    #set views
    gl.view(1+2+4)

    #linewidth to zero
    gl.linewidth(0)
    #set back color
    gl.backcolor(255,255,255)
    #make full screen
    gl.fullscreen(0)
    #make background transparent
    gl.bmptransparent(1)
    #make colorbar transparent
    gl.colorbarcolor(255,255,255,168)
    #change colorbar position: 2-right
    gl.colorbarposition(2)
    #change colorbar size
    gl.colorbarsize(0.1)
    #save image to file
    gl.savebmp(outputfn)

# plot functional ROI masks
for savefn, Tmapfns in Tmapdirs.items():
    if savefn=="Visual_locationrep":
        colorschemes = ["2green","4hot"]
    else:   
        colorschemes = ["4hot"]
    plot_Tmap(Tmapfps   = Tmapfns,
               views    = Tmapviews[savefn],
               outputfn = os.path.join(savedir,savefn+".png"),
               Trange   = Tranges[savefn],
               colorschemes = colorschemes)

