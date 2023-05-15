% The script runs sanity check on the two scanner tasks: navigation task
% and localizer task
% sanity check runs a glm to check stimuli onset and response onset related
% activations.
%
% ------ written by Zillu Liang(2023.4,Oxford)------

clear;clc
%% set up parallel pool
num_workers   = feature('NumCores')-4;
poolobj       =  parpool(num_workers);%set up parallel processing
err_tracker   = struct();
                        
%% sanity checks
glm_name = 'sc_navigation';
err_tracker.(glm_name) = run_glm(glm_name);

glm_name = 'sc_localizer';
err_tracker.(glm_name) = run_glm(glm_name);

%% repetition suppression - 1d location - localizer task
glm_name = 'rs_loc1d_localizer';
err_tracker.(glm_name) = run_glm(glm_name);

%% repetition suppression - 1d location - navigation task
glm_name = 'rs_loc1d_navigation';
err_tracker.(glm_name) = run_glm(glm_name);

%% repetition suppression - feature - localizer task
glm_name = 'rs_f_navigation';
err_tracker.(glm_name) = run_glm(glm_name);

%% close parallel pool
delete(poolobj)

