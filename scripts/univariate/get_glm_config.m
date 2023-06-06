function glm_cofig = get_glm_config(glm_name)
% get the configurations of the glm models
    glms      = glm_gallery;
    glm_names = {glms.name};
    glm_cofig = glms(cellfun(@(x) strcmp(glm_name,x),glm_names));
end

function glms = glm_gallery
    glms = struct('name',{},'filepattern',{},'conditions',{},'contrasts',{});
    
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
    glms(2).modelopt    = struct('use_stick', {true,true});
    glms(2).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(2).contrasts(1).name = 'visual';
    glms(2).contrasts(1).wvec = [1,0];
    glms(2).contrasts(2).name = 'response';
    glms(2).contrasts(2).wvec = [0,1];

    glms(3).name        = 'rs_loc2d_navigation';
    glms(3).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(3).conditions  = {'rstrials','response'};
    glms(3).modelopt    = struct('use_stick', {true,true,false});
    glms(3).pmods       = {{'dist2d'}};
    glms(3).contrasts(1).name = 'euclidean distance';
    glms(3).contrasts(1).wvec = [0,1,0];% weight vector for task regressors
    
    glms(4).name        = 'rs_resploc2d_navigation';
    glms(4).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(4).conditions  = {'rstrials','response'};
    glms(4).modelopt    = struct('use_stick', {true,true,false});
    glms(4).pmods       = {{'dist2d_resp'}};
    glms(4).contrasts(1).name = 'euclidean distance';
    glms(4).contrasts(1).wvec = [0,1,0];% weight vector for task regressors
    
    glms(5).name        = 'rs_loc2d_localizer';
    glms(5).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(5).conditions  = {'rstrials'};
    glms(5).pmods       = {{'dist2d'}};
    glms(5).contrasts(1).name = 'euclidean distance';
    glms(5).contrasts(1).wvec = [0,1];% weight vector for task regressors
    
    glms(6).name = 'traintest_navigation';
    glms(6).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(6).conditions  = {'training','test'};
    glms(6).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(6).contrasts(1).name = 'train_minus_test';
    glms(6).contrasts(1).wvec = [1,-1];% weight vector for task regressors
    glms(6).contrasts(2).name = 'test_minus_train';
    glms(6).contrasts(2).wvec = [-1,1];
    
    allstimid = 0:24;
    glms(7).name = 'LSA_stimuli_navigation';
    glms(7).filepattern = 'sub-.*_task-piratenavigation_run-[1-4]';
    glms(7).conditions  = arrayfun(@(x) sprintf('stim%02d',x),allstimid,'uni',0);
    glms(7).contrasts   = struct('name',{},'type',{},'wvec',{});
   
    trainingstimid = [2,7,10,11,12,13,14,17,22];
    glms(8).name = 'LSA_stimuli_localizer';
    glms(8).filepattern = 'sub-.*_task-localizer_run-[1]';
    glms(8).conditions  = arrayfun(@(x) sprintf('stim%02d',x),trainingstimid,'uni',0);
    glms(8).contrasts   = struct('name',{},'type',{},'wvec',{});    
end