function glm_config = glm_configure(glm_name)
% get the configurations of the glm models
% INPUTS:
% - glm_name: name of the glm as specified in the glm_gallery
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    glms      = glm_gallery;
    glm_names = {glms.name};
    if ~ismember(glm_name,glm_names)
        error("Cannot find glm name in glm gallery, available glms: %s\n", strjoin(glm_names,', '))
    end
    glm_config = glms(cellfun(@(x) strcmp(glm_name,x),glm_names));
end

%% ==============================================================================================
%                                      GLM MODEL DESIGNS
% ==============================================================================================
%  This section is a gallery of GLM configurations for univariate analysis
%  including the ones used to estimate beta series for MVPA.
%  The configurations specify which data columns in the data table should be
%  used as event onsets, event durations and parametric modulators. Weight 
%  vectors and names of contrasts are also specified in the configurations.
%
%  ##
%  Columns in data table of navigation task:
%  'stim_id','stim_img','stim_x','stim_y','stimattr_x','stimattr_y','resp_x','resp_y','resp_dist','respmap_x','respmap_y',...% fields in the orginal data table
%  'onset_stimuli','duration_stimuli',... % stimuli event
%  'onset_response','duration_response',... % response event                
%  'onset_rstrials','duration_rstrials',...  % repetition suppression event
%  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
%  'dist2d','dist2d_resp',... % parametric modulators for repetition suppression:groundtruth distance/recontructed distance from participant response between current stimulus and previous stimulus 
%  'onset_training','duration_training',... % training event   
%  'onset_test','duration_test',... % test event 
%  'onset_stimxx','duration_stimxx', % event for each stimuli
%
% ##
% Columns in data table of localizer task:
%  'stim_id','stim_img','stim_x','stim_y','stimattr_x','stimattr_y','response','acc',...% fields in the orginal data table
%  'dist2d','excluders','rstrials',...% fields in the orginal data table
%  'onset_stimuli',  'duration_stimuli',... % stimuli event
%  'onset_response',... % response event, no duration of response, will be set as stick function in glm                
%  'onset_rstrials', 'duration_rstrials',...  % repetition suppression event
%  'onset_excluders','duration_excluders'
%  'onset_stimxx','duration_stimxx', % event for each stimuli
% ==============================================================================================

function glms = glm_gallery
    glms = struct('name',{}, ...
                  'filepattern',{}, ...
                  'conditions',{}, ...
                  'modelopt',{}, ...
                  'contrasts',{});
    
% ==============================================================================================
% SANITY CHECK MODELS
% ==============================================================================================
    glms(1).name = 'sc_navigation';
    glms(1).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(1).conditions  = {'stimuli','response'};
    glms(1).modelopt    = struct('use_stick', {false,false});
    glms(1).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(1).contrasts(1).name = 'visual';
    glms(1).contrasts(1).wvec = struct('stimuli',{1},'response',{1});%[1,1];% weight vector for task regressors
    glms(1).contrasts(2).name = 'motor';
    glms(1).contrasts(2).wvec = struct('stimuli',{0},'response',{1});%[0,1];
    
    glms(2).name = 'sc_localizer';
    glms(2).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(2).conditions  = {'stimuli','response'};
    glms(2).modelopt    = struct('use_stick', {false,true});
    glms(2).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(2).contrasts(1).name = 'visual';   
    glms(2).contrasts(1).wvec = struct('stimuli',{1},'response',{0});% this is equivalant to: glms(2).contrasts(1).wvec = [1,0]; when there is no empty columns (no empty response run
    glms(2).contrasts(2).name = 'response';
    glms(2).contrasts(2).wvec = struct('stimuli',{0},'response',{1});% this is equivalant to: glms(2).contrasts(2).wvec = [0,1]; 

% ==============================================================================================
% Repetition Suppression MODELS based on Cartesian coordinates 
% ==============================================================================================
    glms(3).name        = 'rs_loc2d_navigation';
    glms(3).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(3).conditions  = {'rstrials','response','excluders'};
    glms(3).modelopt    = struct('use_stick', {true,false,true});
    glms(3).pmods       = {{'dist2d'}};
    glms(3).contrasts(1).name = 'euclidean distance';
    glms(3).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    
    glms(4).name        = 'rs_resploc2d_navigation';
    glms(4).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(4).conditions  = {'rstrials','response','excluders'};
    glms(4).modelopt    = struct('use_stick', {true,false,true});
    glms(4).pmods       = {{'dist2d_resp'}};
    glms(4).contrasts(1).name = 'euclidean distance';
    glms(4).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    
    glms(5).name        = 'rs_loc2d_localizer';
    glms(5).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(5).conditions  = {'rstrials','response','excluders'};
    glms(6).modelopt    = struct('use_stick', {true,true,true});
    glms(5).pmods       = {{'dist2d'}};
    glms(5).contrasts(1).name = 'euclidean distance';
    glms(5).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors: 3 conditions + 1 pmod

% ==============================================================================================
% Difference between train/test
% ==============================================================================================    
    glms(6).name = 'traintest_navigation';
    glms(6).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(6).conditions  = {'training','test','response'};
    glms(6).modelopt    = struct('use_stick', {false,false,false});
    glms(6).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(6).contrasts(1).name = 'train_minus_test';
    glms(6).contrasts(1).wvec = [1,-1,0];% weight vector for task regressors
    glms(6).contrasts(2).name = 'test_minus_train';
    glms(6).contrasts(2).wvec = [-1,1,0];
    
% ==============================================================================================
% LSA glm for extracting beta series - not concatenated
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    glms(7).name = 'LSA_stimuli_navigation';
    glms(7).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(7).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0),{'response'}];
    glms(7).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false}]);
    % F contrast for the overall effect of stimuli
    glms(7).contrasts(1).name = 'stimuli';
    glms(7).contrasts(1).wvec = cell2struct(num2cell(eye(numel(exp.allstim))),... % same as [eye(numel(exp.allstim)),zeros(numel(exp.allstim),1)] 
                                            arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0));
    % separate t contrast for each stimulus
    curr_ccount = numel(glms(7).contrasts);
    for j = 1:numel(exp.allstim)
        glms(7).contrasts(j+curr_ccount).name = sprintf('stim%02d',exp.allstim(j));
        glms(7).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02d',exp.allstim(j)),{1});
    end
    % contrast for each stimulus in odd and even runs
    curr_ccount = numel(glms(7).contrasts);
    for j = 1:numel(exp.allstim)
        idx_odd  = j+curr_ccount;
        idx_even = j+curr_ccount+numel(exp.allstim);
        glms(7).contrasts(idx_odd).name = sprintf('stim%02d_odd',exp.allstim(j));
        glms(7).contrasts(idx_odd).wvec = struct(sprintf('Sn_1_stim%02d',exp.allstim(j)),{1}, ...
                                                 sprintf('Sn_3_stim%02d',exp.allstim(j)),{1});
        glms(7).contrasts(idx_even).name = sprintf('stim%02d_even',exp.allstim(j));
        glms(7).contrasts(idx_even).wvec = struct(sprintf('Sn_2_stim%02d',exp.allstim(j)),{1}, ...
                                                  sprintf('Sn_4_stim%02d',exp.allstim(j)),{1});
    end
   
    glms(8).name = 'LSA_stimuli_localizer';
    glms(8).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(8).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.trainingstim,'uni',0),{'response'}];
    glms(8).modelopt    = struct('use_stick', [repmat({false},size(exp.trainingstim)),{true}]);

% ==============================================================================================
% Neural axis for x/y
% ==============================================================================================  
    glms(9).name = 'axis_loc_navigation'; % location based on ground truth
    glms(9).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(9).conditions  = {'stimuli','response'};
    glms(9).modelopt    = struct('use_stick', {false,false});
    glms(9).pmods       = {{'stim_x','stim_y'}};
    glms(9).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(9).contrasts(1).name = 'stim_x';
    glms(9).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    glms(9).contrasts(2).name = 'stim_y';
    glms(9).contrasts(2).wvec = [0,0,1,0];% weight vector for task regressors

    glms(10).name = 'axis_resploc_navigation'; % location based on ground truth
    glms(10).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(10).conditions  = {'stimuli','response'};
    glms(10).modelopt    = struct('use_stick', {false,false});
    glms(10).pmods       = {{'respmap_x','respmap_y'}};
    glms(10).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(10).contrasts(1).name = 'respmap_x';
    glms(10).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    glms(10).contrasts(2).name = 'respmap_y';
    glms(10).contrasts(2).wvec = [0,0,1,0];% weight vector for task regressors

    glms(11).name = 'axis_loc_localizer'; % location based on ground truth
    glms(11).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(11).conditions  = {'stimuli','response'};
    glms(11).modelopt    = struct('use_stick', {false,true});
    glms(11).pmods       = {{'stim_x','stim_y'}};
    glms(11).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(11).contrasts(1).name = 'stim_x';
    glms(11).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    glms(11).contrasts(2).name = 'stim_y';
    glms(11).contrasts(2).wvec = [0,0,1,0];% weight vector for task regressors

% ==============================================================================================
% LSA glm for extracting beta series - concatenated
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    glms(12).name = 'LSA_stimuli_navigation_concatall';
    glms(12).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(12).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0),{'response'}];
    glms(12).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false}]);
    % contrast for the overall effect of stimuli
    glms(12).contrasts(1).name = 'stimuli';
    glms(12).contrasts(1).wvec = cell2struct(num2cell(eye(numel(exp.allstim))),...
                                             arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0));

    glms(13).name = 'LSA_stimuli_navigation_concatodd';
    glms(13).filepattern = 'sub-.*_task-piratenavigation_run-[1,3]';
    glms(13).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0),{'response'}];
    glms(13).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false}]);
    % contrast for the overall effect of stimuli
    glms(13).contrasts(1).name = 'stimuli';
    glms(13).contrasts(1).wvec = cell2struct(num2cell(ones(numel(exp.allstim),1)),...
                                             arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0));

    glms(14).name = 'LSA_stimuli_navigation_concateven';
    glms(14).filepattern = 'sub-.*_task-piratenavigation_run-[2,4]';
    glms(14).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0),{'response'}];
    glms(14).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false}]);
    % contrast for the overall effect of stimuli
    glms(14).contrasts(1).name = 'stimuli';
    glms(14).contrasts(1).wvec = cell2struct(num2cell(ones(numel(exp.allstim),1)),...
                                             arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0));

% ==============================================================================================
% Difference between train/test
% ==============================================================================================    
    glms(15).name = 'traintest_navigation_wvsworesp';
    glms(15).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(15).conditions  = {'training_resp','test_resp','training_noresp','test_noresp','response'};
    glms(15).modelopt    = struct('use_stick', {true,true,true,true,false});
    glms(15).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(15).contrasts(1).name = 'test_minus_train';
    glms(15).contrasts(1).wvec = [-1,1,-1,1,0];% weight vector for task regressors
    glms(15).contrasts(2).name = 'resp_minus_noresp';
    glms(15).contrasts(2).wvec = [1,1,-1,-1,0];
    glms(15).contrasts(3).name = 'test_minus_train_resp';
    glms(15).contrasts(3).wvec = [-1,1,0,0,0];
    glms(15).contrasts(4).name = 'test_minus_train_noresp';
    glms(15).contrasts(4).wvec = [0,0,-1,1,0];
    glms(15).contrasts(5).name = 'testvstrain_respvsnoresp_interaction';
    glms(15).contrasts(5).wvec = [-1,1,1,-1,0];
    glms(15).contrasts(6).name = 'resp_minus_noresp_run1';
    glms(15).contrasts(6).wvec = struct('Sn_1_training_resp',{1}, ...
                                        'Sn_1_test_resp',{1}, ...
                                        'Sn_1_training_noresp',{-1}, ...
                                        'Sn_1_test_noresp',{-1});
    glms(15).contrasts(7).name = 'resp_minus_noresp_run2';
    glms(15).contrasts(7).wvec = struct('Sn_2_training_resp',{1}, ...
                                        'Sn_2_test_resp',{1}, ...
                                        'Sn_2_training_noresp',{-1}, ...
                                        'Sn_2_test_noresp',{-1});
    glms(15).contrasts(8).name = 'resp_minus_noresp_run3';
    glms(15).contrasts(8).wvec = struct('Sn_3_training_resp',{1}, ...
                                        'Sn_3_test_resp',{1}, ...
                                        'Sn_3_training_noresp',{-1}, ...
                                        'Sn_3_test_noresp',{-1});
    glms(15).contrasts(9).name = 'resp_minus_noresp_run4';
    glms(15).contrasts(9).wvec = struct('Sn_4_training_resp',{1}, ...
                                        'Sn_4_test_resp',{1}, ...
                                        'Sn_4_training_noresp',{-1}, ...
                                        'Sn_4_test_noresp',{-1});
    

% ==============================================================================================
% Repetition Suppression MODELS - Based on features
% ==============================================================================================
    glms(16).name        = 'rs_feacture1d_navigation';
    glms(16).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(16).conditions  = {'rstrials','response','excluders'};
    glms(16).modelopt    = struct('use_stick', {true,false,true});
    glms(16).pmods       = {{'colordist','shapedist'}};
    glms(16).contrasts(1).name = 'same vs diff color';
    glms(16).contrasts(1).wvec = [0,1,0,0,0];
    glms(16).contrasts(2).name = 'same vs diff shape';
    glms(16).contrasts(2).wvec = [0,0,1,0,0];

    glms(17).name        = 'rs_feature2d_navigation';
    glms(17).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(17).conditions  = {'rstrials','response','excluders'};
    glms(17).modelopt    = struct('use_stick', {true,false,true});
    glms(17).pmods       = {{'featuredist'}};
    glms(17).contrasts(1).name = 'feature 2d distance';
    glms(17).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
    
    glms(18).name        = 'rs_color_navigation';
    glms(18).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(18).conditions  = {'rstrials','response','excluders'};
    glms(18).modelopt    = struct('use_stick', {true,false,true});
    glms(18).pmods       = {{'colordist'}};
    glms(18).contrasts(1).name = 'same vs diff color';
    glms(18).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors

    glms(19).name        = 'rs_shape_navigation';
    glms(19).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(19).conditions  = {'rstrials','response','excluders'};
    glms(19).modelopt    = struct('use_stick', {true,false,true});
    glms(19).pmods       = {{'shapedist'}};
    glms(19).contrasts(1).name = 'same vs diff shape';
    glms(19).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
 
% ==============================================================================================
% with/wo response - LSA glm for extracting beta series - not concatenated
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    glms(20).name = 'LSA_stimuli_navigation_wvsworesp';
    glms(20).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(20).conditions  = [arrayfun(@(x) sprintf('stim%02dresp',x),exp.allstim,'uni',0),...
                            arrayfun(@(x) sprintf('stim%02dnoresp',x),exp.allstim,'uni',0),...
                            {'response'}];
    glms(20).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),repmat({false},size(exp.allstim)),{false}]);
    % separate t contrast for each stimulus
    glms(20).contrasts   = struct('name',{},'type',{},'wvec',{});
    curr_ccount = numel(glms(20).contrasts);
    for j = 1:numel(exp.allstim)
        glms(20).contrasts(j+curr_ccount).name = sprintf('stim%02dresp',exp.allstim(j));
        glms(20).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02dresp',exp.allstim(j)),{1});
    end
    curr_ccount = numel(glms(20).contrasts);
    for j = 1:numel(exp.allstim)
        glms(20).contrasts(j+curr_ccount).name = sprintf('stim%02dnoresp',exp.allstim(j));
        glms(20).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02dnoresp',exp.allstim(j)),{1});
    end
% ==============================================================================================
% with/wo response - LSA glm for extracting beta series - concatenated
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    glms(21).name = 'LSA_stimuli_navigation_concatall_wvsworesp';
    glms(21).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(21).conditions  = [arrayfun(@(x) sprintf('stim%02dresp',x),exp.allstim,'uni',0),...
                            arrayfun(@(x) sprintf('stim%02dnoresp',x),exp.allstim,'uni',0),...
                            {'response'}];
    glms(21).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),repmat({false},size(exp.allstim)),{false}]);

    % separate t contrast for each stimulus
    glms(21).contrasts   = struct('name',{},'type',{},'wvec',{});
    curr_ccount = numel(glms(21).contrasts);
    for j = 1:numel(exp.allstim)
        glms(21).contrasts(j+curr_ccount).name = sprintf('stim%02dresp',exp.allstim(j));
        glms(21).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02dresp',exp.allstim(j)),{1});
    end
    curr_ccount = numel(glms(21).contrasts);
    for j = 1:numel(exp.allstim)
        glms(21).contrasts(j+curr_ccount).name = sprintf('stim%02dnoresp',exp.allstim(j));
        glms(21).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02dnoresp',exp.allstim(j)),{1});
    end
% ==============================================================================================
% Train-test Before vs after resp
% ==============================================================================================    
    glms(22).name = 'traintest_navigation_beforevsafterresp';
    glms(22).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(22).conditions  = {'training_beforeresp','test_beforeresp','training_afterresp','test_afterresp','response'};
    glms(22).modelopt    = struct('use_stick', {true,true,true,true,false});
    glms(22).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(22).contrasts(1).name = 'test_minus_train';
    glms(22).contrasts(1).wvec = [-1,1,-1,1,0];% weight vector for task regressors


    glms(22).contrasts(2).name = 'beforeresp_minus_afterresp';
    glms(22).contrasts(2).wvec = [1,1,-1,-1,0];
    glms(22).contrasts(3).name = 'test_minus_train_beforeresp';
    glms(22).contrasts(3).wvec = [-1,1,0,0,0];
    glms(22).contrasts(4).name = 'test_minus_train_afterresp';
    glms(22).contrasts(4).wvec = [0,0,-1,1,0];
    glms(22).contrasts(5).name = 'testvstrain_beforevsafterresp_interaction';
    glms(22).contrasts(5).wvec = [-1,1,1,-1,0];

% ==============================================================================================
% before/after response - LSA glm for extracting beta series - not concatenated
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    glms(23).name = 'LSA_stimuli_navigation_bvsaoresp';
    glms(23).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(23).conditions  = [arrayfun(@(x) sprintf('stim%02dbefore',x),exp.allstim,'uni',0),...
                            {'training_afterresp','test_afterresp'},...,...
                            {'response'}];
    glms(23).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false,false,false}]);
    % separate t contrast for each stimulus
    glms(23).contrasts   = struct('name',{},'type',{},'wvec',{});
    curr_ccount = numel(glms(23).contrasts);
    for j = 1:numel(exp.allstim)
        glms(23).contrasts(j+curr_ccount).name = sprintf('stim%02dbefore',exp.allstim(j));
        glms(23).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02dbefore',exp.allstim(j)),{1});
    end

% ==============================================================================================
% before/after response - LSA glm for extracting beta series - concatenated
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    glms(24).name = 'LSA_stimuli_navigation_concatall_bvsaresp';
    glms(24).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(24).conditions  = [arrayfun(@(x) sprintf('stim%02dbefore',x),exp.allstim,'uni',0),...
                            {'training_afterresp','test_afterresp'},...
                            {'response'}];
    glms(24).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false,false,false}]);

    % separate t contrast for each stimulus
    glms(24).contrasts   = struct('name',{},'type',{},'wvec',{});
    curr_ccount = numel(glms(24).contrasts);
    for j = 1:numel(exp.allstim)
        glms(24).contrasts(j+curr_ccount).name = sprintf('stim%02dbefore',exp.allstim(j));
        glms(24).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02dbefore',exp.allstim(j)),{1});
    end
    
% ==============================================================================================
% Neural axis for x/y with resepct to training location
% ==============================================================================================  
    glms(25).name = 'dist2train_navigation'; % location based on ground truth
    glms(25).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(25).conditions  = {'stimuli','response'};
    glms(25).modelopt    = struct('use_stick', {false,false});
    glms(25).pmods       = {{'x_sign','x_dist','y_sign','y_dist'}};
    glms(25).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(25).contrasts(1).name = 'x_sign';
    glms(25).contrasts(1).wvec = [0,1,0,0,0,0];
    glms(25).contrasts(2).name = 'x_dist';
    glms(25).contrasts(2).wvec = [0,0,1,0,0,0];
    glms(25).contrasts(3).name = 'y_sign';
    glms(25).contrasts(3).wvec = [0,0,0,1,0,0];
    glms(25).contrasts(4).name = 'y_dist';
    glms(25).contrasts(4).wvec = [0,0,0,0,1,0];

% ==============================================================================================
% Neural axis for x/y in localizer with previous trial taken into account
% ==============================================================================================
    glms(26).name = 'axis_loc_wprevtrial_localizer'; % location based on ground truth
    glms(26).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(26).conditions  = {'rstrials','response','excluders'};
    glms(26).modelopt    = struct('use_stick', {false,true,false});
    glms(26).pmods       = {{'stim_x','stim_y','prev_stimx','prev_stimy'}};
    glms(26).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(26).contrasts(1).name = 'stim_x';
    glms(26).contrasts(1).wvec = [0,1,0,0,0,0,0];
    glms(26).contrasts(2).name = 'stim_y';
    glms(26).contrasts(2).wvec = [0,0,1,0,0,0,0];
    glms(26).contrasts(3).name = 'prev_stimx';
    glms(26).contrasts(3).wvec = [0,0,0,1,0,0,0];
    glms(26).contrasts(4).name = 'prev_stimy';
    glms(26).contrasts(4).wvec = [0,0,0,0,1,0,0];

% ==============================================================================================
% Repetition Suppression MODELS - Feature Based
% ==============================================================================================
    glms(27).name        = 'rs_hrchydist_navigation';
    glms(27).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(27).conditions  = {'rstrials','response','excluders'};
    glms(27).modelopt    = struct('use_stick', {true,false,true});
    glms(27).pmods       = {{'hrchydist_ucord','hrchydist_quadr'}};
    glms(27).contrasts(1).name = 'dist - unsigned coordinates';
    glms(27).contrasts(1).wvec = [0,1,0,0,0];
    glms(27).contrasts(2).name = 'dist - quadrants';
    glms(27).contrasts(2).wvec = [0,0,1,0,0];

    glms(28).name        = 'rs_hrchydistucord_navigation';
    glms(28).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(28).conditions  = {'rstrials','response','excluders'};
    glms(28).modelopt    = struct('use_stick', {true,false,true});
    glms(28).pmods       = {{'hrchydist_ucord'}};
    glms(28).contrasts(1).name = 'dist - unsigned coordinates';
    glms(28).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors

    glms(29).name        = 'rs_hrchydistquadr_navigation';
    glms(29).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(29).conditions  = {'rstrials','response','excluders'};
    glms(29).modelopt    = struct('use_stick', {true,false,true});
    glms(29).pmods       = {{'hrchydist_quadr'}};
    glms(29).contrasts(1).name = 'dist - unsigned coordinates';
    glms(29).contrasts(1).wvec = [0,1,0,0];% weight vector for task regressors
   
 % ==============================================================================================
% Neural axis for x/y with resepct to centre for train and test separately
% ==============================================================================================  
    glms(30).name = 'dist2train_navigation_traintest'; % location based on ground truth
    glms(30).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(30).conditions  = {'training','test','response'};
    glms(30).modelopt    = struct('use_stick', {false,false,false});
    glms(30).pmods       = {{'x_sign','x_dist','y_sign','y_dist'},{'x_sign','x_dist','y_sign','y_dist'}};
    glms(30).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(30).contrasts(1).name = 'x_sign_training';
    glms(30).contrasts(1).wvec = [0,1,0,0,0, 0,0,0,0,0, 0];
    glms(30).contrasts(2).name = 'x_dist_training';
    glms(30).contrasts(2).wvec = [0,0,1,0,0, 0,0,0,0,0, 0];
    glms(30).contrasts(3).name = 'y_sign_training';
    glms(30).contrasts(3).wvec = [0,0,0,1,0, 0,0,0,0,0, 0];
    glms(30).contrasts(4).name = 'y_dist_training';
    glms(30).contrasts(4).wvec = [0,0,0,0,1, 0,0,0,0,0, 0];
    glms(30).contrasts(5).name = 'x_sign_test';
    glms(30).contrasts(5).wvec = [0,0,0,0,0, 0,1,0,0,0, 0];
    glms(30).contrasts(6).name = 'x_dist_test';
    glms(30).contrasts(6).wvec = [0,0,0,0,0, 0,0,1,0,0, 0];
    glms(30).contrasts(7).name = 'y_sign_test';
    glms(30).contrasts(7).wvec = [0,0,0,0,0, 0,0,0,1,0, 0];
    glms(30).contrasts(8).name = 'y_dist_test';
    glms(30).contrasts(8).wvec = [0,0,0,0,0, 0,0,0,0,1, 0];


    glms(31).name = 'axis_resploc_navigation_traintest'; % location based on ground truth
    glms(31).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(31).conditions  = {'training','test','response'};
    glms(31).modelopt    = struct('use_stick', {false,false,false});
    glms(31).pmods       = {{'respmap_x','respmap_y'},{'respmap_x','respmap_y'}};
    glms(31).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(31).contrasts(1).name = 'respmap_x_training';
    glms(31).contrasts(1).wvec = [0,1,0, 0,0,0, 0];% weight vector for task regressors
    glms(31).contrasts(2).name = 'respmap_y_training';
    glms(31).contrasts(2).wvec = [0,0,1, 0,0,0, 0];% weight vector for task regressors
    glms(31).contrasts(3).name = 'respmap_x_test';
    glms(31).contrasts(3).wvec = [0,0,0, 0,1,0, 0];% weight vector for task regressors
    glms(31).contrasts(4).name = 'respmap_y_test';
    glms(31).contrasts(4).wvec = [0,0,0, 0,0,1, 0];% weight vector for task regressors
end