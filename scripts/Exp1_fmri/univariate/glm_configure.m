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
%  'stim_id','stim_img','stim_x','stim_y','stim_attrx','stim_attry','resp_x','resp_y',...% fields in the orginal data table
%  'respmap_x','respmap_y',...% response map 
%  'onset_stimuli','duration_stimuli',... % stimuli event          
%  'onset_response','duration_response',... % response event      
%  'onset_rstrials','duration_rstrials',...  % repetition suppression event
%  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
%  'difffeature','diffcolor','diffshape','diffx','diffy','dist2deuc','distx','disty',...% RS parametric modulators for distance
%  'onset_training','duration_training','onset_rstraining','duration_rstraining',... % training event 
%  'onset_test','duration_test','onset_rstest','duration_rstest',... % test event 
%  'onset_stimxx','duration_stimxx', % event for each stimuli
%
% ##
% Columns in data table of localizer task:
%  'stim_id','stim_img','stim_x','stim_y','stim_attrx','stim_attry','response','acc',...% fields in the orginal data table
%  'prev_stimx','prev_stimy',...
%  'onset_stimuli',  'duration_stimuli',... % stimuli event
%  'onset_response',... % response event, no duration of response, will be set as stick function in glm           
%  'onset_rstrials', 'duration_rstrials',...  % repetition suppression event
%  'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
%  'excluders','rstrials',...
%  'diffx','diffy','dist2deuc','distx','disty',...
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
    glms(1).contrasts(3).name = 'resp_min_visual';
    glms(1).contrasts(3).wvec = struct('stimuli',{-1},'response',{1});% this is equivalant to: glms(2).contrasts(2).wvec = [0,1]; 

    glms(2).name = 'sc_localizer';
    glms(2).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(2).conditions  = {'stimuli','response'};
    glms(2).modelopt    = struct('use_stick', {false,true});
    glms(2).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(2).contrasts(1).name = 'visual';   
    glms(2).contrasts(1).wvec = struct('stimuli',{1},'response',{0});% this is equivalant to: glms(2).contrasts(1).wvec = [1,0]; when there is no empty columns (no empty response run
    glms(2).contrasts(2).name = 'response';
    glms(2).contrasts(2).wvec = struct('stimuli',{0},'response',{1});% this is equivalant to: glms(2).contrasts(2).wvec = [0,1]; 
    glms(2).contrasts(3).name = 'resp_min_visual';
    glms(2).contrasts(3).wvec = struct('stimuli',{-1},'response',{1});% this is equivalant to: glms(2).contrasts(2).wvec = [0,1]; 

% ==============================================================================================
% Repetition Suppression MODELS based on Cartesian coordinates 
% ==============================================================================================
    glms(3).name        = 'rs_loc2d_navigation';
    glms(3).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(3).conditions  = {'rstrials','response','excluders'};
    glms(3).modelopt    = struct('use_stick', {false,false,false});
    glms(3).pmods       = {{'dist2deuc'}};
    glms(3).contrasts(1).name = 'euclidean distance';
    glms(3).contrasts(1).wvec = struct('rstrialsxdist2deuc',{1},'excluders',{0},'response',{0});% weight vector for task regressors
    
    glms(4).name        = 'rs_loc2dsepgroup_navigation';
    glms(4).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(4).conditions  = {'rstraining','rstest','rstraintest','response','excluders'};
    glms(4).modelopt    = struct('use_stick', {false,false,false,false,false});
    glms(4).pmods       = {{'dist2deuc'},{'dist2deuc'},{'dist2deuc'}};
    glms(4).contrasts(1).name = 'train2train euclidean distance';
    glms(4).contrasts(1).wvec = struct('rstrainingxdist2deuc',{1});% weight vector for task regressors
    glms(4).contrasts(2).name = 'test2test euclidean distance';
    glms(4).contrasts(2).wvec = struct('rstestxdist2deuc',{1});% weight vector for task regressors
    glms(4).contrasts(3).name = 'train2test euclidean distance';
    glms(4).contrasts(3).wvec = struct('rstraintestxdist2deuc',{1});% weight vector for task regressors
    
    glms(5).name        = 'rs_loc2d_localizer';
    glms(5).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(5).conditions  = {'rstrials','response','excluders'};
    glms(5).modelopt    = struct('use_stick', {false,true,false});
    glms(5).pmods       = {{'dist2deuc'}};
    glms(5).contrasts(1).name = 'euclidean distance';
    glms(5).contrasts(1).wvec = struct('rstrialsxdist2deuc',{1},'excluders',{0},'response',{0});

% ==============================================================================================
% Difference between train/test
% ==============================================================================================    
    glms(6).name = 'traintest_navigation';
    glms(6).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(6).conditions  = {'training','test','response'};
    glms(6).modelopt    = struct('use_stick', {false,false,false});
    glms(6).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(6).contrasts(1).name = 'train_minus_test';
    glms(6).contrasts(1).wvec = struct('training',{1},'test',{-1},'response',{0});% weight vector for task regressors
    glms(6).contrasts(2).name = 'test_minus_train';
    glms(6).contrasts(2).wvec = struct('training',{-1},'test',{1},'response',{0});
    glms(6).contrasts(3).name = 'training';
    glms(6).contrasts(3).wvec = struct('training',{1},'test',{0},'response',{0});
    glms(6).contrasts(4).name = 'test';
    glms(6).contrasts(4).wvec = struct('training',{0},'test',{1},'response',{0});
    glms(6).contrasts(5).name = 'motor';
    glms(6).contrasts(5).wvec = struct('training',{0},'test',{0},'response',{1});
    
% ==============================================================================================
% LSA glm for extracting beta series
% ==============================================================================================  
    exp = get_pirate_defaults(false,'exp');
    n_train = numel(exp.trainingstim);
    n_test  = numel(exp.teststim);
    glms(7).name = 'LSA_stimuli_navigation';
    glms(7).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(7).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0),{'response'}];
    glms(7).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false}]);
    % F contrast for the overall effect of stimuli
    glms(7).contrasts(1).name = 'stimuli';
    glms(7).contrasts(1).wvec = cell2struct(num2cell(eye(numel(exp.allstim))),... 
                                            arrayfun(@(x) sprintf('stim%02d',x), exp.allstim,'uni',0));
    % T contrast for the test and training and difference
    glms(7).contrasts(2).name = 'training';
    glms(7).contrasts(2).wvec = cell2struct(repmat({1/n_train},n_train,1), arrayfun(@(x) sprintf('stim%02d',x),exp.trainingstim,'uni',0));
    glms(7).contrasts(3).name = 'test';
    glms(7).contrasts(3).wvec = cell2struct(repmat({1/n_test}, n_test, 1), arrayfun(@(x) sprintf('stim%02d',x),exp.teststim,'uni',0));
    glms(7).contrasts(4).name = 'train_min_test';
    glms(7).contrasts(4).wvec = cell2struct([repmat({1/n_train},n_train,1); repmat({-1/n_test},n_test,1)], ...
                                            arrayfun(@(x) sprintf('stim%02d',x), [exp.trainingstim exp.teststim],'uni',0));
    glms(7).contrasts(5).name = 'test_min_train';
    glms(7).contrasts(5).wvec = cell2struct([repmat({-1/n_train},n_train,1); repmat({1/n_test},n_test,1)], ...
                                            arrayfun(@(x) sprintf('stim%02d',x), [exp.trainingstim exp.teststim],'uni',0));
    
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
    glms(9).contrasts(1).wvec = struct('stimulixstim_x',{1});
    glms(9).contrasts(2).name = 'stim_y';
    glms(9).contrasts(2).wvec = struct('stimulixstim_y',{1});

    glms(10).name = 'axis_locsepgroup_navigation'; % location based on ground truth
    glms(10).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(10).conditions  = {'training','test','response'};
    glms(10).modelopt    = struct('use_stick', {false,false,false});
    glms(10).pmods       = {{'stim_x','stim_y'},{'stim_x','stim_y'}};
    glms(10).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(10).contrasts(1).name = 'training x';
    glms(10).contrasts(1).wvec = struct('trainingxstim_x',{1});
    glms(10).contrasts(2).name = 'training y';
    glms(10).contrasts(2).wvec = struct('trainingxstim_y',{1});
    glms(10).contrasts(3).name = 'test x';
    glms(10).contrasts(3).wvec = struct('testxstim_x',{1});
    glms(10).contrasts(4).name = 'test y';
    glms(10).contrasts(4).wvec = struct('testxstim_y',{1});
    glms(10).contrasts(5).name = 'train_min_test x';
    glms(10).contrasts(5).wvec = struct('trainingxstim_x',{1},'testxstim_x',{-1});
    glms(10).contrasts(5).name = 'train_min_test y';
    glms(10).contrasts(5).wvec = struct('trainingxstim_x',{-1},'testxstim_y',{1});
    

    glms(11).name = 'axis_loc_localizer'; % location based on ground truth
    glms(11).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(11).conditions  = {'stimuli','response'};
    glms(11).modelopt    = struct('use_stick', {false,true});
    glms(11).pmods       = {{'stim_x','stim_y'}};
    glms(11).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(11).contrasts(1).name = 'stim_x';
    glms(11).contrasts(1).wvec = struct('stimulixstim_x',{1});
    glms(11).contrasts(2).name = 'stim_y';
    glms(11).contrasts(2).wvec = struct('stimulixstim_y',{1});



end