import json
import pandas as pd
import os
from glob import glob
import numpy as np
wk_dir = r'E:\pirate_fmri\Analysis'
data_dir = os.path.join(wk_dir,'data','Exp2_pilotfmri','untidiedbhavior')

# read in rename json to rename and organize bahavior data
## renamer is a dictionary with keys indicating the new subject naming, values indicating [exptid, fMRI dataname]
with open(os.path.join(wk_dir,'data','Exp2_pilotfmri','renamer.json')) as renamer_file:
  renamer = json.load(renamer_file)

sessions = {1:{'name':'pretrain',
               'dir':r'pretrain\json',
               'file_pattern':"cond4_%s_session1.json"},
            2:{'name':'refresher',
               'dir':'refresher',
               'file_pattern':"refresher_%s_run1_*.txt"},
            3:{'name':'scanning',
               'dir':'fmri_behavior',
               'file_pattern':"maintask_%s_run[1-6]_*.txt"},
            4:{'name':'postscan',
               'dir':'postscan',
               'file_pattern':"cond4_%s_session3.json"}
           }

new_outputdir = os.path.join(wk_dir,'data','Exp2_pilotfmri','tidiedbhavior')
keep_columns = ['expt_session','expt_block','expt_trial','expt_task','expt_cond', 
                'expt_coordsys', 'expt_curricula', 'expt_map','ctrl_expt','ctrl_fb','ctrl_resp',
                'stim_id','stim_x', 'stim_y', 'stim_group', 'stim_img',

                'resp_x', 'resp_y','resp_dist', 'resp_rt',
                
                'delay','option_leftlocid', 'option_rightlocid',
                'optionT_locid', 'optionT_x', 'optionT_y', 
                'optionD_locid','optionD_x', 'optionD_y', 
                'resp_choice', 'resp_acc']

for newid in renamer.keys():
   oldids = renamer[newid]
   expt_id = oldids[0]
   subdata = dict()

   #construct filenames
   datafile_path = {
      1:os.path.join(data_dir,sessions[1]['dir'],sessions[1]['file_pattern'] % (expt_id)),
      2:os.path.join(data_dir,sessions[2]['dir'],sessions[2]['file_pattern'] % (expt_id)),
      3:os.path.join(data_dir,sessions[3]['dir'],sessions[3]['file_pattern'] % (expt_id)),
      4:os.path.join(data_dir,sessions[4]['dir'],sessions[4]['file_pattern'] % (expt_id))
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

   xidx = param["slist_fields"]["attrx"]
   yidx = param["slist_fields"]["attry"]
   
   subdata['s1']['stim_attrx'] = [param['stim_list'][id][xidx] for id in subdata['s1']['stim_id']]
   subdata['s1']['stim_attry'] = [param['stim_list'][id][yidx] for id in subdata['s1']['stim_id']]
   subdata['s1']['training'] = subdata['s1']['stim_group'] == "training"
   subdata['s1']['ctrl_expt'] = subdata['s1']['ctrl_ept']
   subdata['s1'] = subdata['s1'].assign(expt_session = 1,ctrl_resp=1)

   ####### Session 2 Refresher #############
   datafile_path[2] = glob(datafile_path[2])[0]
   dataS2 = pd.read_csv(datafile_path[2])
   max_bnums1 = 15
   ntrial_per_refb = 8
   subdata['s2'] = dataS2.assign(training = 1,expt_session = 2,ctrl_expt = True,ctrl_resp = 1,ctrl_fb = True,expt_task=0)
   subdata['s2']['expt_block'] = [int(np.ceil(t/ntrial_per_refb)+max_bnums1) for t in subdata['s2']['expt_trial']]

   ####### Session 3 Scanner #############
   dataS3_Runs = [pd.read_csv(run_fn).assign(run = irun) for run_fn,irun in zip(glob(datafile_path[3]),np.arange(len(glob(datafile_path[3])))+1)]
   max_bnums2 = 23
   dataS3 = pd.concat(dataS3_Runs, axis=0).reset_index(drop=True)
   subdata['s3'] = dataS3.assign(expt_session = 3,ctrl_expt = True,ctrl_fb = False,expt_task=1,ctrl_resp=1)
   subdata['s3']['expt_block'] = [int(run+max_bnums2) for run in subdata['s3']['run']]

   ####### Session 4 post scan #############
   if expt_id!="uPwtCGE97u1E":
      with open(datafile_path[4]) as data_file:
         dataS4 = json.load(data_file)
      subdata['s4'] = pd.DataFrame(dataS4['sdata']).assign(expt_session = 4,ctrl_expt = True,ctrl_fb = False,expt_task=0,ctrl_resp=1)
   else:
      subdata['s4'] = pd.read_csv(os.path.join(data_dir,'postscan','treasurehunt_uPwtCGE97u1E_run1_stt1_10152005.txt')).assign(expt_session = 4,ctrl_expt = True,ctrl_fb = False,expt_task=0,ctrl_resp=1)

   ###concatenate and output to csv
   data = pd.concat([subdata['s1'],subdata['s2'],subdata['s3'],subdata['s4']], axis=0, ignore_index=True)
   allcols = data.columns
   data = data.drop([x for x in allcols if x not in keep_columns], axis=1).assign(subid = newid)
   new_fn = os.path.join(new_outputdir,f'{newid}.csv')
   data.to_csv(new_fn)

valid_ids = renamer.keys()
data_allsub = pd.concat([pd.read_csv(os.path.join(new_outputdir,f'{newid}.csv')) for newid in renamer.keys()], axis=0, ignore_index=True).reset_index(drop=True)
data_allsub.to_csv(os.path.join(wk_dir,'data','Exp2_pilotfmri','all_participants.csv'))