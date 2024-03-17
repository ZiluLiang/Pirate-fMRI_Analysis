import json
import pandas as pd
import os
from glob import glob
import numpy as np
wk_dir = r'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis'
data_dir = os.path.join(wk_dir,'data','untidiedbhavior')

# read in rename json to rename and organize bahavior data
## renamer is a dictionary with keys indicating the new subject naming, values indicating [exptid, fMRI dataname]
with open(os.path.join(wk_dir,'data','renamer.json')) as renamer_file:
  renamer = json.load(renamer_file)

sessions = {1:{'name':'pretrain',
               'dir':r'pretrain\json',
               'file_pattern':"cond4_%s_session1.json"},
            2:{'name':'refresher',
               'dir':'refresher',
               'file_pattern':"refresher_%s_run1_*.txt"},
            3:{'name':'scanning',
               'dir':'fmri_behavior',
               'file_pattern':"maintask_%s_run[1-4]_*.txt"}
           }

new_outputdir = os.path.join(wk_dir,'data','tidiedbhavior')
drop_columns = ['expt_index','ctrl_ept','iti',
                'onset_trial','time_jitter','bonus','start_sx','start_sy',
                'resp_sx','resp_sy','time_arena','time_response','time_fixation','time_cue',
                'resp_rt','resp_acclvl','resp_correct']

for newid,oldids in renamer.items():
   expt_id = oldids[0]
   subdata = dict()

   #construct filenames
   datafile_path = {
      1:os.path.join(data_dir,sessions[1]['dir'],sessions[1]['file_pattern'] % (expt_id)),
      2:os.path.join(data_dir,sessions[2]['dir'],sessions[2]['file_pattern'] % (expt_id)),
      3:os.path.join(data_dir,sessions[3]['dir'],sessions[3]['file_pattern'] % (expt_id))
   }
   paramfile_path = os.path.join(data_dir,r'pretrain\param',"param_%s.json" % (expt_id))

   # get parameter files
   with open(paramfile_path) as param_file:
      param = json.load(param_file)

   # retrive data of different session
   ####### Session 1 Pretrianing ############
   #retrieve data if exact match is found    
   with open(datafile_path[1]) as data_file:
      dataS1 = json.load(data_file)

   dataS1['sdata']['expt_map'] = sum(param['seq_map'][0:param['nb_blocks'][0]],[])    
   subdata['s1'] = pd.DataFrame(dataS1['sdata'])

   subdata['s1']['stim_attrx'] = [param['stim_list'][id][0] for id in subdata['s1']['stim_id']]
   subdata['s1']['stim_attry'] = [param['stim_list'][id][1] for id in subdata['s1']['stim_id']]
   subdata['s1']['ctrl_expt'] = sum(param['seq_eptctrl'][0:param['nb_blocks'][0]],[])
   subdata['s1']['training'] = [1 if ctrl_fb else 0 for ctrl_fb in subdata['s1']['ctrl_fb']]
   subdata['s1'] = subdata['s1'].assign(expt_session = 1,ctrl_resp = 1)

   ####### Session 2 Refresher #############
   datafile_path[2] = glob(datafile_path[2])[0]
   dataS2 = pd.read_csv(datafile_path[2])
   subdata['s2'] = dataS2.assign(training = 1,expt_session = 2,ctrl_expt = True,ctrl_resp = 1,ctrl_fb = True)
   subdata['s2']['expt_block'] = [int(np.ceil(t/9)+17) for t in subdata['s2']['expt_trial']]

   ####### Session 3 Scanner #############
   dataS3_Runs = [pd.read_csv(run_fn).assign(run = irun) for run_fn,irun in zip(glob(datafile_path[3]),np.arange(len(glob(datafile_path[3])))+1)]
   dataS3 = pd.concat(dataS3_Runs)
   subdata['s3'] = dataS3.assign(expt_session = 3,ctrl_expt = True,ctrl_fb = False)
   subdata['s3']['expt_block'] = [int(run+20) for run in subdata['s3']['run']]

   ###concatenate and output to csv
   data = pd.concat([subdata['s1'],subdata['s2'],subdata['s3']], axis=0, ignore_index=True).drop(drop_columns, axis=1).assign(subid = newid)
   new_fn = os.path.join(new_outputdir,f'{newid}.csv')
   data.to_csv(new_fn)

data_allsub = pd.concat([pd.read_csv(os.path.join(new_outputdir,f'{newid}.csv')) for newid in renamer.keys()], axis=0, ignore_index=True)
data_allsub.to_csv(os.path.join(wk_dir,'data','all_participants.csv'))