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
%  Columns in data table of pirate2AFC task:
%    'stim_id','stim_locid','stim_group','stim_left','stim_right','stim_leftgroup','stim_leftgroup','stim_img',...% fields in the orginal data table
%    'stim_x','stim_y','stim_attrx','stim_attry','resp_x','resp_y','resp_choice','resp_acc','resp_rt',...% fields in the orginal data table
%    'stim_map','delay','iti','duration_delay','duration_iti',...
%    'x_dist','x_sign','y_dist','y_sign','hrchydist_ucord','hrchydist_quadr',...% hierachy model: stimuli location wrt screen centre
%    'onset_stimuli', 'duration_stimuli',... % stimuli event
%    'onset_probe',   'duration_probe',... % stimuli event
%    'onset_response','duration_response',... % response event                
%    'onset_rstrials','duration_rstrials',...  % repetition suppression event
%    'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
%    'dist2d','featuredist','leftdist','rightdist',... % parametric modulators for repetition suppression:groundtruth distance/recontructed distance from participant response between current stimulus and previous stimulus 
%    'TD_eucdist','TD_cbdist'
%    'onset_stimxx','duration_stimxx', % event for each stimulus
%    'onset_locxx','duration_locxx', % event for stimuli at eacj locatopm
%    'onset_xx','duration_xx', % event for each stimuli type
%
%
% ##
% Columns in data table of localizer2AFC task:
%    'stim_id','stim_locid','stim_group','stim_left','stim_right','stim_leftgroup','stim_leftgroup','stim_img',...% fields in the orginal data table
%    'stim_x','stim_y','stim_attrx','stim_attry','resp_x','resp_y','resp_choice','resp_acc','resp_rt',...% fields in the orginal data table
%    'stim_map','delay','iti','duration_delay','duration_iti',...
%    'x_dist','x_sign','y_dist','y_sign','hrchydist_ucord','hrchydist_quadr',...% hierachy model: stimuli location wrt screen centre
%    'onset_stimuli', 'duration_stimuli',... % stimuli event
%    'onset_probe',   'duration_probe',... % stimuli event
%    'onset_response','duration_response',... % response event                
%    'onset_rstrials','duration_rstrials',...  % repetition suppression event
%    'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
%    'dist2d','featuredist','leftdist','rightdist',... % parametric modulators for repetition suppression:groundtruth distance/recontructed distance from participant response between current stimulus and previous stimulus 
%    'TD_eucdist','TD_cbdist'
%    'onset_stimxx','duration_stimxx', % event for each stimulus
%    'onset_locxx','duration_locxx', % event for stimuli at eacj locatopm
%    'onset_xx','duration_xx', % event for each stimuli type
%
%
% ## 
% Columns in data table of symboloddball task:
%    'sym_id','sym_side','sym_axloc','sym_map',...% fields in the orginal data table
%    'sym_img','ctrl_resp','resp_choice','isnovel','resp_acc','resp_y','resp_rt',...% fields in the orginal data table
%    'iti','duration_iti',...
%    'onset_stimuli', 'duration_stimuli',... % stimuli event
%    'onset_response',... % response event                
%    'onset_learnedsym','onset_novelsym','duration_learnedsym','duration_novelsym',...
%    'onset_rstrials','duration_rstrials',...  % repetition suppression event
%    'onset_excluders','duration_excluders',...% excluded trials in repetition suppression
%    'dist2d','distmap','distPTAaxis','dist1dPTAaxloc','dist5dPTAaxloc'
% ==============================================================================================

function glms = glm_gallery
    glms = struct('name',{}, ...
                  'filepattern',{}, ...
                  'conditions',{}, ...
                  'modelopt',{}, ...
                  'contrasts',{});
    
% ==============================================================================================
% Repetition Suppression MODELS based on Cartesian coordinates and PTA
% ==============================================================================================
    glms(1).name        = 'rs_allpmods_symboloddball';
    glms(1).filepattern = 'sub-.*_task-symboloddball_run-[1-2]';
    glms(1).conditions  = {'rstrials','response','excluders'};
    glms(1).modelopt    = struct('use_stick', {false,true,false});
    glms(1).pmods       = {{'dist2d','distmap','distPTAaxis','dist1dPTAaxloc','dist5dPTAaxloc'}};
    glms(1).contrasts(1).name = 'euclidean distance';
    glms(1).contrasts(1).wvec = struct('rstrialsdist2d',{1});
    glms(1).contrasts(2).name = 'diffmap';
    glms(1).contrasts(2).wvec = struct('rstrialsdistmap',{1});
    glms(1).contrasts(3).name = 'diffaxis';
    glms(1).contrasts(3).wvec = struct('rstrialsdistPTAaxis',{1});
    glms(1).contrasts(4).name = 'dist1dPTAaxloc';
    glms(1).contrasts(4).wvec = struct('rstrialsdist1dPTAaxloc',{1});
    glms(1).contrasts(5).name = 'dist5dPTAaxloc';
    glms(1).contrasts(5).wvec = struct('rstrialsdist5dPTAaxloc',{1});

% ==============================================================================================
% Novel vs Learn symbols in symbol odd ball
% ==============================================================================================
    glms(2).name        = 'novellearned_symboloddball';
    glms(2).filepattern = 'sub-.*_task-symboloddball_run-[1-2]';
    glms(2).conditions  = {'learnedsym','novelsym','response'};
    glms(2).modelopt    = struct('use_stick', {false,false,true});
    glms(2).contrasts(1).name = 'learned_minus_novel';
    glms(2).contrasts(1).wvec = struct('learnedsym',{1},'novelsym',{-1});% this is better than using weight vector because some participant didn't respond
    glms(2).contrasts(2).name = 'novel_minus_learned';
    glms(2).contrasts(2).wvec = struct('learnedsym',{-1},'novelsym',{1});% weight vector for task regressors
    glms(2).contrasts(3).name = 'visual';
    glms(2).contrasts(3).wvec = struct('learnedsym',{1},'novelsym',{1});
    glms(2).contrasts(4).name = 'motor';
    glms(2).contrasts(4).wvec = struct('response',{1});
% ==============================================================================================
% Difference between train/test
% ==============================================================================================    
    glms(3).name = 'stimulitype_pirate2AFC';
    glms(3).filepattern = 'sub-.*_task-pirate2AFC_run-[1-6]';
    glms(3).conditions  = {'training','validation','testcenter','TMtestnoncenter','CMtestnoncenter','response'};
    glms(3).modelopt    = struct('use_stick', {false,false,false,false,false,true});
    glms(3).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(3).contrasts(1).name = 'test_minus_train';
    glms(3).contrasts(1).wvec = struct('training',{-1},'validation',{1/4},'testcenter',{1/4},'TMtestnoncenter',{1/4},'CMtestnoncenter',{1/4});% weight vector for task regressors
    glms(3).contrasts(2).name = 'noncenter>center';
    glms(3).contrasts(2).wvec = struct('training',{-1/2},'validation',{1/3},'testcenter',{-1/2},'TMtestnoncenter',{1/3},'CMtestnoncenter',{1/3});% weight vector for task regressors
    glms(3).contrasts(3).name = 'NCcrossmap_minus_trainmap';
    glms(3).contrasts(3).wvec = struct('validation',{-1/2},'TMtestnoncenter',{-1/2},'CMtestnoncenter',{1});
    glms(3).contrasts(4).name = 'visual';
    glms(3).contrasts(4).wvec = struct('training',{1},'validation',{1},'testcenter',{1},'TMtestnoncenter',{1},'CMtestnoncenter',{1});
    glms(3).contrasts(5).name = 'motor';
    glms(3).contrasts(5).wvec = struct('response',{1});

% % ==============================================================================================
% % LSA glm for extracting beta series - not concatenated
% % ==============================================================================================  
%     exp = get_pirate_defaults(false,'exp');
%     glms(7).name = 'LSA_stimuli_pirate2AFC';
%     glms(7).filepattern = 'sub-.*_task-pirate2AFC_run-[1-4]';
%     glms(7).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0),{'response'}];
%     glms(7).modelopt    = struct('use_stick', [repmat({false},size(exp.allstim)),{false}]);
%     % F contrast for the overall effect of stimuli
%     glms(7).contrasts(1).name = 'stimuli';
%     glms(7).contrasts(1).wvec = cell2struct(num2cell(eye(numel(exp.allstim))),... % same as [eye(numel(exp.allstim)),zeros(numel(exp.allstim),1)] 
%                                             arrayfun(@(x) sprintf('stim%02d',x),exp.allstim,'uni',0));
%     % separate t contrast for each stimulus
%     curr_ccount = numel(glms(7).contrasts);
%     for j = 1:numel(exp.allstim)
%         glms(7).contrasts(j+curr_ccount).name = sprintf('stim%02d',exp.allstim(j));
%         glms(7).contrasts(j+curr_ccount).wvec = struct(sprintf('stim%02d',exp.allstim(j)),{1});
%     end
%     % contrast for each stimulus in odd and even runs
%     curr_ccount = numel(glms(7).contrasts);
%     for j = 1:numel(exp.allstim)
%         idx_odd  = j+curr_ccount;
%         idx_even = j+curr_ccount+numel(exp.allstim);
%         glms(7).contrasts(idx_odd).name = sprintf('stim%02d_odd',exp.allstim(j));
%         glms(7).contrasts(idx_odd).wvec = struct(sprintf('Sn_1_stim%02d',exp.allstim(j)),{1}, ...
%                                                  sprintf('Sn_3_stim%02d',exp.allstim(j)),{1});
%         glms(7).contrasts(idx_even).name = sprintf('stim%02d_even',exp.allstim(j));
%         glms(7).contrasts(idx_even).wvec = struct(sprintf('Sn_2_stim%02d',exp.allstim(j)),{1}, ...
%                                                   sprintf('Sn_4_stim%02d',exp.allstim(j)),{1});
%     end
% 
%     glms(8).name = 'LSA_stimuli_localizer';
%     glms(8).filepattern = 'sub-.*_task-localizer_run-[1]';
%     glms(8).conditions  = [arrayfun(@(x) sprintf('stim%02d',x),exp.trainingstim,'uni',0),{'response'}];
%     glms(8).modelopt    = struct('use_stick', [repmat({false},size(exp.trainingstim)),{true}]);


end