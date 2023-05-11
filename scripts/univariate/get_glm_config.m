function glm_cofig = get_glm_config(glm_name)
    glms      = glm_gallery;
    glm_names = {glms.name};
    glm_cofig = glms(cellfun(@(x) strcmp(glm_name,x),glm_names));
end

function glms = glm_gallery
    glms = struct('name',{},'filepattern',{},'conditions',{},'contrasts',{});
% TODO: add contrast specification glm.contrast(

    glms(1).name = 'sc_navigation';
    glms(1).filepattern = 'sub-%s_task-piratenavigation_run-[1-4]';
    glms(1).conditions  = {'stimuli','response'};
    glms(1).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(1).contrasts(1).name = 'stimuli';
    glms(1).contrasts(1).type = 't';
    glms(1).contrasts(1).wvec  = gen_contrast_vec([1,0],6,4);
    glms(1).contrasts(2).name = 'response';
    glms(1).contrasts(2).type = 't';
    glms(1).contrasts(2).wvec  = gen_contrast_vec([0,1],6,4);

    glms(2).name = 'sc_localizer';
    glms(2).filepattern = 'sub-%s_task-localizer_run-[1]';
    glms(2).conditions = {'stimuli','response'};
    glms(2).contrasts   = struct('name',{},'type',{},'wvec',{});
    glms(2).contrasts(1).name = 'stimuli';
    glms(2).contrasts(1).type = 't';
    glms(2).contrasts(1).wvec  = gen_contrast_vec([1,0],6,4);
    glms(2).contrasts(2).name = 'response';
    glms(2).contrasts(2).type = 't';
    glms(2).contrasts(2).wvec  = gen_contrast_vec([0,1],6,4);

    

end

function vec = gen_contrast_vec(v,nN,nS)
% generate a 1 x ((nC+nN)*nS + nS) contrast vector
% INPUTS:
% v  - contrast in one session
% nC - number of conditions
% nN - number of nuisance regressors
% based on spm convention that regressors are ordered as in:
% [C-1_sess-1,C-2_sess-1,...N-1_sess-1,N-2_sess-1,...C-nC_sess-N,...N-nN_sess-nS,S-1,...S-nS]
% where C is condition regressor, N is nuisance regressor and S is the session regressor

    if nargin<4, nN = 6; end
    v = v/nS; % scaled by number of sessions
    v = [v,zeros(1,nN)];
    % concatenate across session
    vec = [repmat(v,1,nS),zeros(1,nS)];

end