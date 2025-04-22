from copy import deepcopy
import os
import gl


import warnings
warnings.simplefilter('ignore', category=FutureWarning)

project_path = r'E:\pirate_fmri\Analysis'
fmridata_dir = os.path.join(project_path,'data','Exp1_fmri','fmri')
ROIRSAdir = os.path.join(fmridata_dir,'ROIRSA','AALandHCPMMP1andFUNCcluster')

def plot_masks(maskfps,views,outputfn):
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
    gl.view(1+2+4)
    
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
    #save image to file
    gl.savebmp(outputfn)

# plot functional ROI masks
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
anatmask_dir = deepcopy(ROIRSAdir)
anatmask_fns = {"HPC":["HPC_bilateral"],
                "HPC_L":["HPC_left"],
                "HPC_R":["HPC_right"],
                "V1":["V1_bilateral"],
                "V1_L":["V1_left"],
                "V1_R":["V1_right"],
                "V2":["V2_bilateral"],
                "V2_L":["V2_left"],
                "V2_R":["V2_right"],
                "vmPFC":["vmPFC_bilateral"],
                "vmPFC_L":["vmPFC_left"],
                "vmPFC_R":["vmPFC_right"]
                }
anatmask_view = {
    "HPC":[26,-18,0],
    "HPC_L":[-26,-18,0],
    "HPC_R":[26,-18,0],
    "V1":[12,-72,8],
    "V1_L":[-12,-72,8],
    "V1_R":[12,-72,8],
    "V2":[-6,-81,16],
    "V2_L":[-6,-81,16],
    "V2_R":[6,-81,16],
    "vmPFC":[-5,54,-2],
    "vmPFC_L":[-5,54,-2],
    "vmPFC_R":[5,54,-2]
}
for savefn, maskfns in anatmask_fns.items():
    plot_masks(maskfps=[os.path.join(anatmask_dir,maskfn+".nii") for maskfn in maskfns],
               views = anatmask_view[savefn],
               outputfn = os.path.join(anatmask_dir,"maskvisualize",savefn+".png"))