clear;
clc;
participant = 'sub001';
taskname = 'localizer';
data_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\fmri\data\fmri';
output_dir = fullfile(data_dir,'datacheckingGLM',taskname,participant);
preproc_dir = fullfile(data_dir,'converted',participant);
runs = [5];
% load regressor specification
% regressor should be a struct of size (n_session,1) and each field
% corresponds to an event.
% regressor = struct('cue',struct('onset',a numeric array of event onset,...
%                                    'duration',a numeric array of event duration,...),...
%                    'response',struct('onset',[1,4,7,...],...
%                                    'duration',[1,1,1,...]))
regressors = dataloader();

spec2est_1stlvl(preproc_dir,output_dir,runs,regressors)