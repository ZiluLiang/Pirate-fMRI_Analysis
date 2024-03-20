import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

# paths
study_dir = r"E:\pirate_fmri\Analysis\data\Exp2_pilots\1_M1_V1\data"
data_dir = r"E:\pirate_fmri\Analysis\data\Exp2_pilots\1_M1_V1\data\json"
param_dir = r"E:\pirate_fmri\Analysis\data\Exp2_pilots\1_M1_V1\data\param"

# output bonus for payment
dataset = []
for fn in os.listdir(data_dir):
    with open(os.path.join(data_dir,fn)) as f:
        dataset.append(json.load(f))

bonus_filepath=os.path.join(
    r"D:\Dropbox\OnlineStudies\pirate_symbol\M1\data",
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

# check the bonus file to select participants who finished both days to get valid id
valid_exptid = ["7bt9uIz5G1aJ",
                "DMGY9zCfGDYx",
                "HgUrUWFTRiIE",
                "Ij4NO2rW8SbI",
                "jIb8kZKzSFNc",
                "jYuzKuOlKAV1",
                "lKo186FW3Hk0",
                "mSZwQS13pi4K",
                "NSnEI5v2eoeR",
                "TbSUHibinOv3"
                ]

# organize task data
dataset = []
data_dfs = []
for id in valid_exptid:
    data_files = [os.path.join(data_dir,f"cond4_{id}_session1.json"),os.path.join(data_dir,f"cond4_{id}_session2.json")]
    param_file = os.path.join(param_dir,f"param_{id}.json")
    subdfs = []
    for fn in data_files:
        with open(os.path.join(fn)) as f:
            data = json.load(f)
            dataset.append(data)
            subdfs.append(pd.DataFrame(data["sdata"]).assign(prolificid=data['edata']['expt_turker'],subid=id))
    subdf = pd.concat(subdfs).reset_index(drop=True)
    data_dfs.append(subdf)
    with open(param_file) as f:
            param = json.load(f)

org_data_dfs = []
for subdf in data_dfs:
    subdf["istraining"] = [np.logical_xor(x==0,y==0) for x,y in zip(subdf["stim_x"],subdf["stim_y"])] 
    ts_df_s1 = subdf[(subdf.ctrl_ept==1)&(subdf.expt_task==0)&(subdf.expt_session==0)].copy()
    ts_df_s2 = subdf[(subdf.ctrl_ept==1)&(subdf.expt_task==0)&(subdf.expt_session==1)].copy()
    stims = {"train":[[],[]],
                "test":[[],[]]}
    for b,b_df in ts_df_s1.groupby("expt_block"):
        if np.shape(b_df[b_df.istraining])[0]==8:
            stims["train"][0].append(np.unique(b_df[b_df.istraining].stim_id))
        elif np.shape(b_df[~b_df.istraining])[0]==8:
            stims["test"][0].append(np.unique(b_df[~b_df.istraining].stim_id))

    for b,b_df in ts_df_s2.groupby("expt_block"):
        if np.shape(b_df[b_df.istraining])[0]>0:
            stims["train"][1].append(np.unique(b_df[b_df.istraining].stim_id))
        if np.shape(b_df[~b_df.istraining])[0]>0:
            stims["test"][1].append(np.unique(b_df[~b_df.istraining].stim_id))    
            import itertools
    #if want to double check
    #np.all([np.array_equal(a1,a2) for a1,a2 in itertools.combinations(stims["test"][0],2)])
    #np.all([np.array_equal(a1,a2) for a1,a2 in itertools.combinations(stims["test"][1],2)])
    #np.all([np.array_equal(a1,a2) for a1,a2 in itertools.combinations(stims["train"][0],2)])

    trainid= np.unique(stims["train"][0])
    testid_in_s1 = np.unique(stims["test"][0])
    testid_in_s2 =  np.unique(stims["test"][1])[[(x not in testid_in_s1) for x in np.unique(stims["test"][1])]]
    group_name = np.array(["training","validation","test"])
    subdf["stim_group"] = [group_name[[x in trainid, x in testid_in_s1, x in testid_in_s2]][0] for x in subdf.stim_id]
    org_data_dfs.append(subdf)

org_data = pd.concat(org_data_dfs).reset_index(drop=True).fillna(value=np.nan)
print(org_data.dtypes)

bool2int_columns = ["ctrl_fb","ctrl_ept","istraining"]
for k in bool2int_columns:
    org_data[k] = org_data[k].astype(int)
print(org_data.dtypes)

org_data["resp_correct"] = org_data["resp_correct"].astype(int)

print(org_data.dtypes)
org_data.to_csv(os.path.join("D:\Dropbox\OnlineStudies\pirate_symbol\M1","pilot_data.csv"))


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
        cluster_task_data = data['ddata'].pop("cluster")
        cluster_data_dfs.append(pd.DataFrame(cluster_task_data).assign(prolificid=data['edata']['expt_turker'],subid=id))
        debrief_data_dfs.append(pd.DataFrame(data['ddata'],index=[k]).assign(prolificid=data['edata']['expt_turker'],subid=id))

cluster_df = pd.concat(cluster_data_dfs).reset_index(drop=True)
debrief_df = pd.concat(debrief_data_dfs).reset_index(drop=True)
cluster_df.to_csv(os.path.join(study_dir,"cluster_data.csv"))
debrief_df.to_csv(os.path.join(study_dir,"debrief_data.csv"))    
