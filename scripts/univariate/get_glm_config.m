function glm_cofig = get_glm_config(glm_name)
    glms      = glm_gallery;
    glm_names = {glms.name};
    glm_cofig = glms(cellfun(@(x) strcmp(glm_name,x),glm_names));
end

function glms = glm_gallery
    glms = struct('name',{},'filepattern',{},'conditions',{},'contrasts',{});
    nN = 12; % number of nuisance regressors
    
    glms(1).name = 'sc_navigation';
    glms(1).filepattern = 'sub-%s_task-piratenavigation_run-[1-4]';
    glms(1).conditions  = {'stimuli','response'};
    glms(1).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(1).contrasts(1).name = 'visual';
    glms(1).contrasts(1).type = 't';
    glms(1).contrasts(1).wvec  = gen_contrast_vec([1,1],nN,4);% weight vector for task regressors
    glms(1).contrasts(2).name = 'motor';
    glms(1).contrasts(2).type = 't';
    glms(1).contrasts(2).wvec  = gen_contrast_vec([0,1],nN,4);

    glms(2).name = 'sc_localizer';
    glms(2).filepattern = 'sub-%s_task-localizer_run-[1]';
    glms(2).conditions = {'stimuli','response'};
    glms(2).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(2).contrasts(1).name = 'visual';
    glms(2).contrasts(1).type = 't';
    glms(2).contrasts(1).wvec  = gen_contrast_vec([1,0],nN,1);
    glms(2).contrasts(2).name = 'response';
    glms(2).contrasts(2).type = 't';
    glms(2).contrasts(2).wvec  = gen_contrast_vec([0,1],nN,1);

    glms(3).name = 'rs_f_navigation';
    glms(3).filepattern = 'sub-%s_task-piratenavigation_run-[1-4]';
    glms(3).conditions  = {'samecolor','diffcolor','sameshape','diffshape','excluderp'};
    glms(3).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(3).contrasts(1).name = 'same_minus_diff_color';
    glms(3).contrasts(1).type = 't';
    glms(3).contrasts(1).wvec  = gen_contrast_vec([1,-1,0,0,0],nN,4);% weight vector for task regressors
    glms(3).contrasts(2).name = 'same_minus_diff_shape';
    glms(3).contrasts(2).type = 't';
    glms(3).contrasts(2).wvec  = gen_contrast_vec([0,0,1,-1,0],nN,4);
    
    glms(4).name = 'rs_loc1d_navigation';
    glms(4).filepattern = 'sub-%s_task-piratenavigation_run-[1-4]';
    glms(4).conditions  = {'samex','diffx','samey','diffy','excluderp'};
    glms(4).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(4).contrasts(1).name = 'same_minus_diff_x';
    glms(4).contrasts(1).type = 't';
    glms(4).contrasts(1).wvec = gen_contrast_vec([1,-1,0,0,0],nN,4);% weight vector for task regressors
    glms(4).contrasts(2).name = 'same_minus_diff_y';
    glms(4).contrasts(2).type = 't';
    glms(4).contrasts(2).wvec = gen_contrast_vec([0,0,1,-1,0],nN,4);
    
    glms(5).name = 'rs_loc1d_localizer';
    glms(5).filepattern = 'sub-%s_task-localizer_run-[1]';
    glms(5).conditions  = {'samex','diffx','samey','diffy'};
    glms(5).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(5).contrasts(1).name = 'same_minus_diff_x';
    glms(5).contrasts(1).type = 't';
    glms(5).contrasts(1).wvec = gen_contrast_vec([1,-1,0,0],nN,1);% weight vector for task regressors
    glms(5).contrasts(2).name = 'same_minus_diff_y';
    glms(5).contrasts(2).type = 't';
    glms(5).contrasts(2).wvec = gen_contrast_vec([0,0,1,-1],nN,1);
    
end

function vec = gen_contrast_vec(v,nN,nS)
% generate contrast vector
% based on spm convention that regressors are ordered as in:
% [C-1_sess-1,C-2_sess-1,...N-1_sess-1,N-2_sess-1,...C-nC_sess-N,...N-nN_sess-nS,S-1,...S-nS]
% where C is condition regressor, N is nuisance regressor and S is the session regressor% INPUTS:
% v  - contrast in one session
% nS - number of sessions
% nN - number of nuisance regressors

    v = v/nS; % scaled by number of sessions
    v = [v,zeros(1,nN)];
    % concatenate across session
    vec = [repmat(v,1,nS),zeros(1,nS)];

end