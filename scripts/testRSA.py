import itertools
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from multivariate.dataloader import ActivityPatternDataLoader
from multivariate.helper import compute_rdm,checkdir,lower_tri,upper_tri
from multivariate.rsa_estimator import PatternCorrelation,MultipleRDMRegression
import pandas as pd
import glob

import numpy
import matplotlib

class ModelRDM:
    """ set up the model rdms

    Parameters
    ----------
    stimid : numpy.ndarray
        a 1D numpy array of stimuli ids
    stimloc : numpy.ndarray
        a 2D numpy array of stimuli locations (x and y)
    stimfeature : numpy.ndarray
        a 2D numpy array of stimuli features (color and feature)
    n_session : int, optional
        number of sessions, by default 1
    split_sess : bool, optional
        split the rdm into within-session and between-session or not, by default True
    """
    
    def __init__(self,
                 stimid:numpy.ndarray,
                 stimloc:numpy.ndarray,
                 stimfeature:numpy.ndarray,
                 stimgroup:numpy.ndarray,
                 n_session:int=1):
        self.n_session   = n_session
        self.n_stim      = len(stimid)
        self.stimid      = numpy.tile(stimid,(n_session,1))
        self.stimloc     = numpy.tile(stimloc,(n_session,1))
        self.stimfeature = numpy.tile(stimfeature,(n_session,1))
        self.stimgroup   = numpy.tile(stimgroup,(n_session,1))
        models = {"loc2d":self.euclidean2d(),
                  "loc1dx":self.euclidean1d(0),
                  "loc1dy":self.euclidean1d(1),
                  "feature2d":self.feature2d(),
                  "feature1dx":self.feature1d(0),
                  "feature1dy":self.feature1d(1),
                  "stimuli":self.identity(),
                  "stimuligroup":self.identity(self.stimgroup)
                }

        # split into sessions
        if n_session>1:
            BS = self.session() # 0 - within session; 1 - within session
            WS = 1 - BS         # 0 - between session; 1 - between session
            BS[BS==0]=numpy.nan
            WS[WS==0]=numpy.nan
            tmp = list(models.items())
            for k,v in tmp:
                ws_n  = 'within_'+k
                rdmws = numpy.multiply(v,WS)
                bs_n  = 'between_'+k
                rdmbs = numpy.multiply(v,BS)
                models |= {ws_n:rdmws,bs_n:rdmbs}
            models["session"] = self.session()
        self.models = models

    def __str__(self):
        return 'The following model rdms are created:\n' + ',\n'.join(
            self.models.keys()
        )

    def session(self)->numpy.ndarray:
        """calculate model rdm based on session, if the pair is in the same session, distance will be 0, otherwise will be 1
        
        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        S = numpy.zeros((self.n_stim,self.n_stim)) # matrix for same session
        D = numpy.ones((self.n_stim,self.n_stim)) # matrix for different session
        
        r = []
        for j in range(self.n_session):
            c = [D for _ in range(self.n_session)]
            c[j] = S
            r.append(c)
        modelrdm = numpy.block(r)
        return modelrdm
    
    def euclidean2d(self)->numpy.ndarray:
        """calculate model rdm based on 2d euclidean distance.
        
        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        modelrdm = compute_rdm(self.stimloc,metric="euclidean")
        return modelrdm

    def euclidean1d(self,dim)->numpy.ndarray:
        """calculate model rdm based on 1d euclidean distance.

        Parameters
        ----------
        dim : int
        which axis (0:x, 1:y) should be used to calculated rdm.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        X,Y = numpy.meshgrid(self.stimloc[:,dim],self.stimloc[:,dim])
        modelrdm = abs(X-Y)
        return modelrdm

    def feature2d(self)->numpy.ndarray:
        """calculate model rdm based on both stimuli features, if the pair:
            shares 0 feature  - sqrt(2);
            shares 1 feature  - 1;
            shares 2 features - 0;

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        RDM_attrx = self.feature1d(0)
        RDM_attry = self.feature1d(1)
        modelrdm  = numpy.sqrt(RDM_attrx+RDM_attry)
        return modelrdm
    
    def feature1d(self,dim:int)->numpy.ndarray:
        """calculate model rdm based on 1 stimuli feature, if the pair has the same feature, distance will be zero, otherwise will be one.

        Parameters
        ----------
        dim : int
        which feature should be used to calculated rdm.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        X,Y = numpy.meshgrid(self.stimfeature[:,dim],self.stimfeature[:,dim])
        modelrdm = 1. - abs(X==Y) # if same feature, distance=0
        return modelrdm
    

    def identity(self,identity_arr=None)->numpy.ndarray:
        """calculate model rdm based on stimuli identity, if the pair is the same stimuli, distance will be zero, otherwise will be one.

        Returns
        -------
        numpy.ndarray
            2D numpy array of model rdm
        """
        if identity_arr is None:
            X,Y = numpy.meshgrid(self.stimid,self.stimid)
        else:
            X,Y = numpy.meshgrid(identity_arr,identity_arr)
        modelrdm = 1. - abs(X==Y)# if same stimuli, distance=0
        return modelrdm
    
    def visualize(self,modelname:str or list="all",tri:int=0,annot:bool=False)->matplotlib.figure:
        """plot model rdms using seaborn heatmap

        Parameters
        ----------
        modelname : str or list of strings, optional
            the name of model rdm to be plotted, by default "all"
        tri: int, optional
            show lower(0), upper(1) or the whole matrix. lower and upper will exclude diagonal elements by default 0
        annot: bool, optional
            show the value in the heatmap, by default False

        Returns
        -------
        matplotlib.figure
            the plotted figure
        """
        if isinstance(modelname,list):
            if numpy.all([m in self.models.keys() for m in modelname]):
                plot_models = modelname
            else:
                plot_models = list(self.models.keys())
        elif isinstance(modelname,str):
            if modelname in self.models.keys():
                plot_models = [modelname]
            elif modelname == "all":
                plot_models = list(self.models.keys())
            else:
                print("invalid model name, plotting all models")
                plot_models = list(self.models.keys())

        n_model = len(plot_models)
        n_row = int(numpy.sqrt(n_model))
        n_col = int(numpy.ceil(n_model/n_row))
        fig,axes = plt.subplots(n_row,n_col,figsize = (5*n_col, 5*n_row))
        for j,k in enumerate(plot_models):
            v = numpy.full_like(self.models[k],numpy.nan)
            if tri==0:
                _,idx = lower_tri(v)
            elif tri==1:
                _,idx = upper_tri(v)
            elif tri==2:
                idx = numpy.where(numpy.isnan(v))
            v[idx] = self.models[k][idx]

            if n_model>1:
                sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85},annot=annot)
                axes.flatten()[j].set_title(k)
            else:
                sns.heatmap(v,ax=axes,square=True,cbar_kws={"shrink":0.85},annot=annot)
                axes.set_title(k)
        if n_row*n_col>1:
            for k in numpy.arange(numpy.size(axes.flatten())-1-j)+1:
                fig.delaxes(axes.flatten()[j+k])
        return fig

project_path = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
fmri_output_path = os.path.join(project_path,'data','fmri')
glm_name = 'LSA_stimuli_navigation'

preprocess = ["smoothed5mmLSA","unsmoothedLSA"]
with open(os.path.join(project_path,'scripts','pirate_defaults.json')) as f:
    pirate_defaults = json.load(f)
    subid_list = pirate_defaults['participants']['validids']
n_run = 4

for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)
    beta_flist4r = []
    beta_flistoe = []
    fmask_flist = []
    pmask_flist = []
    run_stim_labels = []
    y_dict = {"id":[],
            "image":[],
            "locx":[],
            "locy":[],
            "color":[],
            "shape":[],
            "training":[]} 

    for subid in subid_list:
        print(f"retrieving data from {subid}")

        # load stimuli list
        stim_list_fn = glob.glob(os.path.join(fmri_output_path,'beh',subid,'sub*_stimlist.txt'))[0]
        stim_list =  pd.read_csv(stim_list_fn, sep=",", header=0).sort_values(by = ['stim_id'], ascending=True,inplace=False)
        # get stimuli id
        stim_id = np.array(stim_list['stim_id'])
        # get stimuli image
        stim_image = np.array([x.replace('.png','') for x in stim_list["stim_img"]])
        # get 2d location
        stim_locx = np.array(stim_list['stim_x'])
        stim_locy = np.array(stim_list['stim_y'])
        # get visual features
        stim_color = np.array([x.replace('.png','').split('_')[0] for x in stim_list["stim_img"]])
        stim_shape = np.array([x.replace('.png','').split('_')[1] for x in stim_list["stim_img"]])
        # get training/test stimuli classification
        stim_train = np.array(stim_list['training'])

        # build list of beta maps
        firstlvl_dir = os.path.join(LSA_GLM_dir,'first',subid)

        y_dict["id"].append(stim_id)
        y_dict["image"].append(stim_image)
        y_dict["locx"].append(stim_locx)
        y_dict["locy"].append(stim_locy)
        y_dict["color"].append(stim_color)
        y_dict["shape"].append(stim_shape)
        y_dict["training"].append(stim_train)

        beta_flist4r.append(os.path.join(firstlvl_dir,'stimuli_4r.nii'))    
        beta_flistoe.append(os.path.join(firstlvl_dir,'stimuli_oe.nii'))    
        fmask_flist.append(os.path.join(firstlvl_dir,'mask.nii'))
        pmask_flist.append(os.path.join(firstlvl_dir,'reliability_mask.nii'))

analysis = {
    # image-based rdm
    "sc_betweens_stimuli":   ['between_stimuli'],
    "sc_alls_stimuli":       ['stimuli'],
    # feature-based: color(x) or shape(y)
    "sc_withins_feature1d":  ['within_feature1dx','within_feature1dy'],
    "sc_betweens_feature1d": ['between_feature1dx','between_feature1dy'],
    "sc_alls_feature1d":     ['feature1dx','feature1dy'],
    # map-based
    "betweens_loc2d":  ["between_loc2d"],
    "withins_loc2d":   ["within_loc2d"],
    "alls_loc2d":      ["loc2d"],
    "betweens_loc1d":  ["between_loc1dx","between_loc1dy"],
    "withins_loc1d":   ["within_loc1dx","within_loc1dy"],
    "alls_loc1d":      ["loc1dx","loc1dy"]
    }
anatmaskdir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks\anat'

anat_roi = ["hippocampus","parahippocampus","occipital","ofc"]
laterality = ["left","right","bilateral"]

beta_flist = {"fourruns":beta_flist4r,
              "oddeven":beta_flistoe}
n_sess = {"fourruns":4,
          "oddeven":2}
mask_flist = {"noselection":fmask_flist,
              "reliability_ths0":pmask_flist}

def run_ROIRSA(beta_img, mask_imgs, regression_models, regressor_names,subid, group=None):
    APD = ActivityPatternDataLoader(beta_img,mask_imgs)
    activitypattern = APD.X
    for k in group:
        group_mean_pervoxel = np.mean(activitypattern[np.where(group == k),:],1)
        activitypattern[np.where(group == k),:] = activitypattern[np.where(group == k),:] - group_mean_pervoxel
    result = dict(zip(regression_models.keys(),[[]] * len(regression_models.keys())))
    for k,v in regression_models.items():
        MR = MultipleRDMRegression(compute_rdm(activitypattern,'correlation'),v)
        MR.fit()
        result[k] = MR.result
    df = []
    for k, v in regressor_names.items():
        reg_name_dict = dict(zip(range(len(v)),v))
        res_df = pd.DataFrame(result[k]).T.rename(columns = reg_name_dict).assign(analysis=k,subid=subid)
        df.append(res_df)
    return pd.concat(df,axis=0)

# preprocess - ds - vselect - roi -laterality
all_df = []
for p in preprocess: 
    LSA_GLM_dir = os.path.join(fmri_output_path,p,glm_name)
    for ds_name, ds in beta_flist.items():
        for vselect in mask_flist:            
            result = dict(zip(analysis.keys(),[[]] * len(analysis.keys())))            
            for j,subid in enumerate(subid_list):
                model_rdm = ModelRDM(stimid = y_dict["image"][j],
                                        stimloc = np.vstack([y_dict["locx"][j],y_dict["locy"][j]]).T,
                                        stimfeature = np.vstack([y_dict["color"][j],y_dict["shape"][j]]).T,
                                        stimgroup = y_dict["training"][j],
                                    n_session=n_sess[ds_name])
                reg_models = {k: [model_rdm.models[m] for m in v] for k, v in analysis.items()}
                for roi, lat in itertools.product(anat_roi, laterality):
                    print(f"{p} - {ds_name} - {vselect} - {subid} - {roi} = {lat}")
                    anat_mask = os.path.join(anatmaskdir,f'{roi}_{lat}.nii')
                    subriolat_df = run_ROIRSA(beta_img = ds[j],
                                              mask_imgs = [mask_flist[vselect][j],anat_mask],
                                              regression_models = reg_models,
                                              regressor_names=analysis,
                                              subid=subid,
                                              group=numpy.tile(y_dict["training"][j],(n_sess[ds_name],)))
                    subriolat_df = subriolat_df.assign(roi = roi, laterality = lat, voxselect = vselect, ds = ds_name, preprocess=p)
                    all_df.append(subriolat_df)

ROIRSA_output_path = os.path.join(fmri_output_path,'ROIRSA',glm_name+'separatecentering')
checkdir(ROIRSA_output_path)
df = pd.concat(all_df,axis=0)
df.to_csv(os.path.join(ROIRSA_output_path,'roirsa.csv'))

id_cols = ["ds","preprocess","voxselect","roi","laterality",'subid','analysis']
for k,v in analysis.items():
    plot_df = df.loc[df["analysis"]==k,tuple(id_cols+v)]
    plot_df["roi_l"] = plot_df[['roi', 'laterality']].apply(lambda x: '_'.join(x), axis=1)
    plot_df["ds_preproc"] = plot_df[['ds', 'preprocess']].apply(lambda x: '_'.join(x), axis=1)
    plot_df = pd.melt(plot_df, id_vars=id_cols+["roi_l","ds_preproc"], value_vars=v)
    g = sns.catplot(data=plot_df, x="roi_l", y="value",
    hue="voxselect",col="variable",row="ds_preproc",
    kind="box", aspect=3,sharex=True,sharey=True)
    tmp = [plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
    g.savefig(os.path.join(ROIRSA_output_path,f'{k}.png'))