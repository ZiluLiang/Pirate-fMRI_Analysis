import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

# paths
study_dir = r"E:\pirate_fmri\Analysis\data\Exp2_pilots\2_M2_V1"
data_dir  =  os.path.join(study_dir,"data","json")
param_dir =  os.path.join(study_dir,"data","param")

##################################### SOME PARTICIPANTS DATA ARE NOT SAVED ###################################
#data for 73ToiYjyUPlD and 6Wgfkt0TDbfN are in tmp
for id in ['73ToiYjyUPlD','6Wgfkt0TDbfN']:
    txtdat_fn = os.path.join(study_dir,"data","session2","tmp","cond_4",f"{id}.txt")
    f = open(txtdat_fn)
    txtdata = f.readlines()
    txtdata[0] = txtdata[0].replace('sdata ','{"sdata" ') + "}"
    txtdata[1] = txtdata[1].replace('edata ','{"edata" ') + "}"
    txtdata[2] = txtdata[2].replace('ddata ','{"ddata" ') + "}"
    txtdata[3] = txtdata[3].replace('parameters ','{"parameters" ') + "}"
    #json.loads(','.join(txtdata))
    #','.join(txtdata)
    data = {}
    for k,t in zip(["sdata","edata","ddata","parameters"],txtdata):
        data[k]=json.loads(t)[k]
    if "bonus_pound" not in data["edata"].keys():
        data["edata"]["bonus_pound"] = data["edata"]["bonus"]/100   
    with open(os.path.join(data_dir,f"cond4_{id}_session2.json"), 'w') as f:
        json.dump(data,f)

#data for lqUwIrHBBAUP participant is in tmp, but bonus is known
id='lqUwIrHBBAUP'
txtdat_fn = os.path.join(study_dir,"data","session2","tmp","cond_4",f"{id}.txt")
f = open(txtdat_fn)
txtdata = f.readlines()
txtdata[0] = txtdata[0].replace('sdata ','{"sdata" ') + "}"
txtdata[1] = txtdata[1].replace('edata ','{"edata" ') + "}"
txtdata[2] = txtdata[2].replace('ddata ','{"ddata" ') + "}"
txtdata[3] = txtdata[3].replace('parameters ','{"parameters" ') + "}"
#json.loads(','.join(txtdata))
#','.join(txtdata)
data = {}
for k,t in zip(["sdata","edata","ddata","parameters"],txtdata):
    data[k]=json.loads(t)[k]
data["edata"]["bonus_pound"] = 9
with open(os.path.join(data_dir,f"cond4_{id}_session2.json"), 'w') as f:
    json.dump(data,f)

#two participants' data is incomplete 
#data for iH2bagqLtDnj and vqih6rQGsv6x is in resize
for id in ['iH2bagqLtDnj','vqih6rQGsv6x']:
    txtdat_fn = os.path.join(study_dir,"data","session2","resize","cond_4",f"{id}.txt")
    f = open(txtdat_fn)
    txtdata = f.readlines()
    txtdata[0] = txtdata[0].replace('sdata ','{"sdata" ') + "}"
    txtdata[1] = txtdata[1].replace('edata ','{"edata" ') + "}"
    txtdata[2] = txtdata[2].replace('ddata ','{"ddata" ') + "}"
    txtdata[3] = txtdata[3].replace('parameters ','{"parameters" ') + "}"
    #json.loads(','.join(txtdata))
    #','.join(txtdata)
    data = {}
    for k,t in zip(["sdata","edata","ddata","parameters"],txtdata):
        data[k]=json.loads(t)[k]
    if "bonus_pound" not in data["edata"].keys():
        data["edata"]["bonus_pound"] = data["edata"]["bonus"]/100    
    with open(os.path.join(data_dir,f"cond4_{id}_session2.json"), 'w') as f:
        json.dump(data,f)
##############################################################################################################

##################################### OUTPUT BONUS FOR PAYMENT ###################################
dataset = []
for fn in os.listdir(data_dir):
    with open(os.path.join(data_dir,fn)) as f:
        dataset.append(json.load(f))

bonus_filepath=os.path.join(
    study_dir,
    "bonus.csv"
)
bonus_df = pd.DataFrame(
    np.vstack([[data["edata"]["expt_turker"] for data in dataset],
[data["edata"]["bonus_pound"] for data in dataset],
[data["edata"]["expt_subject"] for data in dataset],
[np.unique(data["sdata"]["expt_session"])[0] for data in dataset]]).T,
columns=["prolificid","bonus","expt_id","session"]
)
(bonus_df[~bonus_df.prolificid.str.contains("test")]).to_csv(bonus_filepath)
##############################################################################################################


# check the bonus file to select participants who finished both days to get valid id
valid_exptid = [
    "0x3b1d7RNwtn",
    "4i2wCqKlzfcz",
    "6Wgfkt0TDbfN",
    "73ToiYjyUPlD",

    "ei39tqQQ2Uoj",
    "IaUzcq81FpYy",
    "j5qR4VPBKbAD",
    "XeoSBLkhOwc5",

    "7X9sxTIl0sFc",
    "zrWMqvmHKVSV",
    "Tf8ODdWcZp2M",
    "YBQSartY8cGi",

    "NnGwayWijsWN" ## seems to miss some data?
    #"lqUwIrHBBAUP", ## data incomplete
    #"iH2bagqLtDnj", ## data incomplete
    #"vqih6rQGsv6x"  ## data incomplete
]



######################################## CHECK THE LAYOUT OF THE STIMULI #######################################
#### training stim
ncol = len(valid_exptid)#min([5,len(valid_exptid)])
nrow = 3#int(np.ceil(len(valid_exptid)/ncol))
stimsplit_fig,stimsplit_axes = plt.subplots(nrow,ncol,figsize=(ncol*2,nrow*2))
style_dict = {"training":"P","validation":"s","test":"D"}

for jsub,id in enumerate(valid_exptid):
    param_file = os.path.join(param_dir,f"param_{id}.json")
    with open(param_file) as f:
        param = json.load(f)
        stim_list = param["stim_list"]

    stimdf=pd.DataFrame(stim_list,
                        columns=list(param["slist_fields"].keys()))
    stimdf["maptype"] = ["training maps" if map in [0,1] else "cross maps" for map in np.array(stimdf["map"])]

    tmap_filter = (stimdf.maptype=="training maps")
    trtim_filter = (stimdf.stimuligroup=="training")    
    vstim_filter = (stimdf.stimuligroup=="validation")    
    testim_filter = (stimdf.stimuligroup=="test")
    ax1,ax2,ax3 = stimsplit_axes[:,jsub]
    sns.scatterplot(
        stimdf[tmap_filter&trtim_filter],
        x="attrx",y="attry",
        hue="map",
        style="stimuligroup",markers = style_dict,
        ax=ax1)
    ax1.set_title(id)
    sns.scatterplot(
        stimdf[tmap_filter&vstim_filter],
        x="attrx",y="attry",
        hue="map",
        style="stimuligroup",markers = style_dict,
        ax=ax2)
    sns.scatterplot(
        stimdf[tmap_filter&testim_filter],
        x="attrx",y="attry",
        hue="map",
        style="stimuligroup",markers = style_dict,
        ax=ax3)
    
map_handles,map_labels = ax1.get_legend_handles_labels()[0][:3],ax1.get_legend_handles_labels()[1][:3]
sg_handles = sum([ax.get_legend_handles_labels()[0][4:] for ax in [ax1,ax2,ax3]], [])
sg_labels = sum([ax.get_legend_handles_labels()[1][4:] for ax in [ax1,ax2,ax3]],[])
sg_handles = [ax1.get_legend_handles_labels()[0][3]] + sg_handles
sg_labels  = [ax1.get_legend_handles_labels()[1][3]] + sg_labels

stimsplit_fig.legend(map_handles+sg_handles,map_labels+sg_labels,
                     ncol=len(sg_labels)+len(map_labels),loc="upper center",bbox_to_anchor=(0.5,1))
[ax.legend().remove() for ax in stimsplit_axes.flatten() if ax.legend is not None]

stimsplit_fig.suptitle("stimuli types in different maps",y=1.05)
stimsplit_fig.tight_layout()
stimsplit_fig.savefig(os.path.join(study_dir,"stimuli layout.png"),bbox_inches="tight")
#######################################################################################################################

# organize task data
dataset = []
data_dfs = []
for id in valid_exptid:
    data_files = [os.path.join(data_dir,f"cond4_{id}_session1.json"),os.path.join(data_dir,f"cond4_{id}_session2.json")]
    #data_files = [os.path.join(data_dir,f"cond4_{id}_session1.json")]
    param_file = os.path.join(param_dir,f"param_{id}.json")
    with open(param_file) as f:
            param = json.load(f)
            stim_list = param["stim_list"]
    subdfs = []
    for fn in data_files:
        with open(os.path.join(fn)) as f:
            data = json.load(f)
            dataset.append(data)
            subdfs.append(pd.DataFrame(data["sdata"]).assign(prolificid=data['edata']['expt_turker'],subid=id))
    subdf = pd.concat(subdfs).reset_index(drop=True)
    subdf["istraining"] = subdf["stim_group"] == "training"
    data_dfs.append(subdf)

org_data = pd.concat(data_dfs).reset_index(drop=True).fillna(value=np.nan)
print(org_data.dtypes)

bool2int_columns = ["ctrl_fb","ctrl_ept","istraining"]
for k in bool2int_columns:
    org_data[k] = org_data[k].astype(int)
print(org_data.dtypes)

org_data["resp_correct"] = org_data["resp_correct"].astype(int)

print(org_data.dtypes)
org_data.to_csv(os.path.join(study_dir,"task_data.csv"))


# organize debrief and clusters data
dataset = []
cluster_data_dfs = []
debrief_data_dfs = []
for k,id in enumerate(valid_exptid):
    fn = os.path.join(data_dir,f"cond4_{id}_session2.json")
    param_file = os.path.join(param_dir,f"param_{id}.json")
    with open(os.path.join(fn)) as f:
        data = json.load(f)
        dataset.append(data)
        if "cluster" in data['ddata'].keys():
            cluster_task_data = data['ddata'].pop("cluster")
            cluster_data_dfs.append(pd.DataFrame(cluster_task_data).assign(prolificid=data['edata']['expt_turker'],subid=id))
            debrief_data_dfs.append(pd.DataFrame(data['ddata'],index=[k]).assign(prolificid=data['edata']['expt_turker'],subid=id))
        else: 
            print(f"{id} is missing cluster data")

cluster_df = pd.concat(cluster_data_dfs).reset_index(drop=True)
debrief_df = pd.concat(debrief_data_dfs).reset_index(drop=True)
cluster_df.to_csv(os.path.join(study_dir,"cluster_data.csv"))
debrief_df.to_csv(os.path.join(study_dir,"debrief_data.csv"))    
