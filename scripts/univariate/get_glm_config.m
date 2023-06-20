function glm_cofig = get_glm_config(glm_name)
% get the configurations of the glm models
    glms      = glm_gallery;
    glm_names = {glms.name};
    glm_cofig = glms(cellfun(@(x) strcmp(glm_name,x),glm_names));
end

%% glms designs
% ==============================================================================================
% columns in data table of navigation task:
% 'stim_id','stim_img','stim_x','stim_y','start_x','start_y','resp_x','resp_y','resp_dist','respmap_x','respmap_y',...% fields in the orginal data table
% 'onset_stimuli','duration_stimuli',... % stimuli event
% 'onset_response','duration_response',... % response event                
% 'onset_rstrials','duration_rstrials',...  % repetition suppression event
% 'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
% 'dist2d','dist2d_resp',... % parametric modulators for repetition suppression:groundtruth distance/recontructed distance from participant response between current stimulus and previous stimulus 
% 'onset_training','duration_training',... % training event   
% 'onset_test','duration_test',... % test event 
% 'onset_stimxx','duration_stimxx', % event for each stimuli
% ==============================================================================================
% columns in data table of localizer task:
% 'stim_id','stim_img','stim_x','stim_y','response','acc',...% fields in the orginal data table
% 'dist2d','excluders','rstrials',...% fields in the orginal data table
% 'onset_stimuli',  'duration_stimuli',... % stimuli event
% 'onset_response',... % response event, no duration of response, will be set as stick function in glm                
% 'onset_rstrials', 'duration_rstrials',...  % repetition suppression event
% 'onset_excluders','duration_excluders'
% 'onset_stimxx','duration_stimxx', % event for each stimuli
% ==============================================================================================

function glms = glm_gallery
    glms = struct('name',{},'filepattern',{},'conditions',{},'modelopt',{},'contrasts',{});
    
    glms(1).name = 'sc_navigation';
    glms(1).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(1).conditions  = {'stimuli','response'};
    glms(1).modelopt    = struct('use_stick', {false,false});
    glms(1).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(1).contrasts(1).name = 'visual';
    glms(1).contrasts(1).wvec = [1,1];% weight vector for task regressors
    glms(1).contrasts(2).name = 'motor';
    glms(1).contrasts(2).wvec = [0,1];
    
    glms(2).name = 'sc_localizer';
    glms(2).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(2).conditions  = {'stimuli','response'};
    glms(2).modelopt    = struct('use_stick', {false,true});
    glms(2).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(2).contrasts(1).name = 'visual';
    glms(2).contrasts(1).wvec = [1,0];
    glms(2).contrasts(2).name = 'response';
    glms(2).contrasts(2).wvec = [0,1];

    glms(3).name        = 'rs_loc2d_navigation';
    glms(3).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(3).conditions  = {'rstrials','response','excluders'};
    glms(3).modelopt    = struct('use_stick', {false,false,false});
    glms(3).pmods       = {{'dist2d'}};
    glms(3).contrasts(1).name = 'euclidean distance';
    glms(3).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    
    glms(4).name        = 'rs_resploc2d_navigation';
    glms(4).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(4).conditions  = {'rstrials','response','excluders'};
    glms(4).modelopt    = struct('use_stick', {false,false,false});
    glms(4).pmods       = {{'dist2d_resp'}};
    glms(4).contrasts(1).name = 'euclidean distance';
    glms(4).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    
    glms(5).name        = 'rs_loc2d_localizer';
    glms(5).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(5).conditions  = {'rstrials','response','excluders'};
    glms(6).modelopt    = struct('use_stick', {false,true,false});
    glms(5).pmods       = {{'dist2d'}};
    glms(5).contrasts(1).name = 'euclidean distance';
    glms(5).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors: 3 conditions + 1 pmod
    
    glms(6).name = 'traintest_navigation';
    glms(6).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(6).conditions  = {'training','test','response'};
    glms(6).modelopt    = struct('use_stick', {false,false,false});
    glms(6).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(6).contrasts(1).name = 'train_minus_test';
    glms(6).contrasts(1).wvec = [1,-1,0];% weight vector for task regressors
    glms(6).contrasts(2).name = 'test_minus_train';
    glms(6).contrasts(2).wvec = [-1,1,0];
    
    allstimid = 0:24;
    glms(7).name = 'LSA_stimuli_navigation';
    glms(7).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(7).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),allstimid,'uni',0),{'response'}];
    glms(7).modelopt    = struct('use_stick', [repmat({false},size(allstimid)),{false}]);
   
    trainingstimid = [2,7,10,11,12,13,14,17,22];
    glms(8).name = 'LSA_stimuli_localizer';
    glms(8).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(8).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),trainingstimid,'uni',0),{'response'}];
    glms(8).modelopt    = struct('use_stick', [repmat({false},size(trainingstimid)),{true}]);

    glms(9).name = 'axis_loc_navigation'; % location based on ground truth
    glms(9).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(9).conditions  = {'stimuli','response'};
    glms(9).modelopt    = struct('use_stick', {false,false});
    glms(9).pmods       = {{'stim_y','stim_x'}};
    glms(9).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(9).contrasts(1).name = 'stim_x';
    glms(9).contrasts(1).wvec = [1,0,0];% weight vector for task regressors
    glms(9).contrasts(2).name = 'stim_y';
    glms(9).contrasts(2).wvec = [0,1,0];% weight vector for task regressors

    glms(10).name = 'axis_resploc_navigation'; % location based on ground truth
    glms(10).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(10).conditions  = {'stimuli','response'};
    glms(10).modelopt    = struct('use_stick', {false,false});
    glms(10).pmods       = {{'respmap_x','respmap_y'}};
    glms(10).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(10).contrasts(1).name = 'respmap_x';
    glms(10).contrasts(1).wvec = [1,0,0];% weight vector for task regressors
    glms(10).contrasts(2).name = 'respmap_y';
    glms(10).contrasts(2).wvec = [0,1,0];% weight vector for task regressors

    glms(11).name = 'axis_loc_localizer'; % location based on ground truth
    glms(11).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(11).conditions  = {'stimuli','response'};
    glms(11).modelopt    = struct('use_stick', {false,false});
    glms(11).pmods       = {{'stim_x','stim_y'}};
    glms(11).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(11).contrasts(1).name = 'stim_x';
    glms(11).contrasts(1).wvec = [1,0,0];% weight vector for task regressors
    glms(11).contrasts(2).name = 'stim_y';
    glms(11).contrasts(2).wvec = [0,1,0];% weight vector for task regressors

end