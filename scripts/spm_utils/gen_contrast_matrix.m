function w_mat = gen_contrast_matrix(subSPM,weights,flag_rescale)
% generate contrast vector
% INPUTS:
%  - subSPM: path to first level SPM.mat file or the loaded SPM struct;
%  - weights can be one of the following:
%    (1) contrast matrix in one session: this is convenient if there are
%    same number of regressors per session.
%    or when the regressors differs across sessions, it is convenient to pass:
%    (2) a struct specifying the weights of a regressor, with regressor names
%       or pattern as fieldnames.  Unspecified regressors will be assigned
%       a weight of zero.
%  - flag_rescale: whether or not to rescale the weights depending on the number of regressors
%      if flag_rescale is true, the weights will be divided by the number of regressors found
%      if flag_rescale is false, the weights will stay as specified in weights
% -----------------------------------------------------------------------    
% Author: Zilu Liang

%TODO: build contrast with regressor names. 
% current script is a SUPER BAD solution to deal with constraints in struct field naming, need to come up with cleaner solution

    if ischar(subSPM)
        if exist(subSPM,'file')
            subSPM = load(subSPM).SPM;
        else
            error('SPM file do not exists')
        end
    else
        if ~isstruct(subSPM)
            error('first input must be full path to SPM.mat file or the loaded SPM struct')
        end
    end

    if nargin<3, flag_rescale=true;end

    if isnumeric(weights)
        nS = numel(subSPM.nscan);
        nuisance_idx = find_regressor_idx(subSPM,'Sn(1) R');
        nN = numel(nuisance_idx);        
        w_mat = contrast_matrix(weights,nN,nS,flag_rescale);
    elseif isstruct(weights)
        weighted_regressors = fieldnames(weights);
        %TODO:
        % This is a SUPER BAD solution to deal with constraints in struct field naming, need to come up with cleaner solution
        reg_pattern = cellfun(@(x) regexpPattern(strrep(x,'_','.*')),weighted_regressors,'uni',0);
        if ~all(cellfun(@(x) any(contains(subSPM.xX.name,x)),reg_pattern))
            error('the following regressors are not found in the model: %s', ...
                strjoin(weighted_regressors(cellfun(@(x) ~ismember(x,subSPM.xX.name),reg_pattern)),', ') ...
                )
        end        
        w_mat = zeros(numel(weights),numel(subSPM.xX.name));
        for j = 1:numel(weighted_regressors)
            reg_idx = find_regressor_idx(subSPM,reg_pattern{j});
            w_col   = reshape([weights.(weighted_regressors{j})],numel(weights),1) ./ (numel(reg_idx)^flag_rescale);
            w_mat(:,reg_idx) = repmat(w_col,1,numel(reg_idx));
        end
    end    
end

function w_mat = contrast_matrix(wm,nN,nS,flag_rescale)
% generate contrast vector
% based on spm convention that regressors are ordered as in:
% [C-1_sess-1,C-2_sess-1,...N-1_sess-1,N-2_sess-1,...C-nC_sess-N,...N-nN_sess-nS,S-1,...S-nS]
% where C is condition regressor, N is nuisance regressor and S is the session regressor
% INPUTS:
% wm  - contrast matrix in one session
% nN  - number of nuisance regressors
% nS  - number of sessions
    wm        = wm./(nS^flag_rescale);
    nC        = size(wm,1);
    sess_wmat = [wm,zeros(nC,nN)];
    w_mat     = [repmat(sess_wmat,1,nS),zeros(nC,nS)];
end
