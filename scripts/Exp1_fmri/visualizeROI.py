"""
This script is used to visualize the anatomical masks used for the ROI-based analyses.
Note that it cannot be ran directly in the python console, but should be ran in MRIcroGL.
"""

from copy import deepcopy
import os
import gl


import warnings
warnings.simplefilter('ignore', category=FutureWarning)

project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIdata')

def plot_masks(maskfps,views,outputfn,multiview=True,view=1):
    #open background image
    gl.loadimage("spm152")
    gl.minmax(0,10,80)
    
    #open overlays:
    for j, maskfp in enumerate(maskfps):
        gl.overlayload(maskfp)
        gl.minmax(j+1,0,1)
    
    #set view coordinates
    gl.orthoviewmm(*views)
    
    #set views
    if multiview:
        gl.view(1+2+4)
    else:
        gl.view(view) # Display Axial (1), Coronal (2), Sagittal (4), Flipped Sagittal (8), MPR (16), Mosaic (32) or Rendering (64)
    
    #set color palette
    gl.colorname(1,"8redyell")
    gl.colorname(0,"Grayscale")

    #linewidth to zero
    gl.linewidth(0)
    #set back color
    gl.backcolor(255,255,255)
    #make full screen
    gl.fullscreen(1)
    #make background transparent
    gl.bmptransparent(1)
    #make colorbar transparent
    gl.colorbarcolor(255,255,255,168)
    #change colorbar position: 2-right
    gl.colorbarposition(2)

    # mask with backgroud
    gl.overlaymaskwithbackground(1)
    #save image to file
    gl.savebmp(outputfn)

# plot functional ROI masks
plot_functional = False
if plot_functional:
    funcmask_dir = deepcopy(ROIRSAdir)
    funcmask_fns = {
        "testlocParietalSup":['testgtlocParietalSup_bilateral'],
        "testlocParietalSup_L":['testgtlocParietalSup_left'],
        "testlocParietalSup_R":['testgtlocParietalSup_right'],
        "testlocTPMid": ['testgtlocTPMid_bilateral'],
        "testlocTPMid_L": ['testgtlocTPMid_left'],
        "testlocTPMid_R": ['testgtlocTPMid_right']
    }
    funcmask_views = {
        "testlocParietalSup":[12,-57,60],
        "testlocParietalSup_L":[12,-57,60],
        "testlocParietalSup_R":[12,-57,60],
        "testlocTPMid": [-34,14,-34],
        "testlocTPMid_L": [-34,14,-34],
        "testlocTPMid_R": [-34,14,-34]
    }

    for savefn, maskfns in funcmask_fns.items():

        plot_masks(maskfps=[os.path.join(funcmask_dir,maskfn+".nii") for maskfn in maskfns],
                views = funcmask_views[savefn],
                outputfn = os.path.join(funcmask_dir,savefn+".png"))

# plot anatomical ROI masks
plot_anatomical = True
if plot_anatomical:
    anatmask_dir = deepcopy(ROIRSAdir)
    anatmask_fns = {"PPC":["PPC_bilateral"],
                    "V1":["V1_bilateral"],
                    "HPC":["HPC_bilateral"],
                    "vmPFC":["vmPFC_bilateral"]
                    }
    anatmask_view = {
        "PPC":[22.5,-61,56],
        "V1":[12,-72,10],
        "HPC":[26,-18,0],
        "vmPFC":[-5,54,-2]
    }
    for savefn, maskfns in anatmask_fns.items():
        plot_masks(maskfps=[os.path.join(anatmask_dir,maskfn+".nii") for maskfn in maskfns],
                views = anatmask_view[savefn],
                outputfn = os.path.join(anatmask_dir,"finalfigs","maskvisualize",savefn+".png"))
        
    for savefn, maskfns in anatmask_fns.items():
        plot_masks(maskfps=[os.path.join(anatmask_dir,maskfn+".nii") for maskfn in maskfns],
                   views = anatmask_view[savefn],
                   multiview=False,view=4,
                   outputfn = os.path.join(ROIRSAdir,"finalfigs","maskvisualize",f"sagital_{savefn}.png"))