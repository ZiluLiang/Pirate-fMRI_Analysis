function w_mat = gen_contrast_matrix(v,nBF,nN,nS)
% generate contrast vector
% based on spm convention that regressors are ordered as in:
% [C-1_sess-1,C-2_sess-1,...N-1_sess-1,N-2_sess-1,...C-nC_sess-N,...N-nN_sess-nS,S-1,...S-nS]
% where C is condition regressor, N is nuisance regressor and S is the session regressor
% INPUTS:
% v  - contrast in one session
% nBF - number of basis functions
% nN  - number of nuisance regressors
% nS  - number of sessions
% TODO: generate contrasts based on regressor names?
    v = v/nS;
    nC = numel(v);
    sess_wmat = nan(nBF,nC*nBF+nN);
    for j = 1:nBF
        bF_wvec = arrayfun(@(w) [zeros(1,j-1) w zeros(1,nBF-j)],v,'uni',0);
        sess_wmat(j,:) = [cat(2,bF_wvec{:}), zeros(1,nN)];        
    end
    w_mat = [repmat(sess_wmat,1,nS),zeros(nBF,nS)];
end

% function vec = gen_contrast_vec(v,nN,nS)
% % generate contrast vector
% % based on spm convention that regressors are ordered as in:
% % [C-1_sess-1,C-2_sess-1,...N-1_sess-1,N-2_sess-1,...C-nC_sess-N,...N-nN_sess-nS,S-1,...S-nS]
% % where C is condition regressor, N is nuisance regressor and S is the session regressor% INPUTS:
% % v  - contrast in one session
% % nS - number of sessions
% % nN - number of nuisance regressors
% 
%     v = v/nS; % scaled by number of sessions
%     v = [v,zeros(1,nN)];
%     % concatenate across session
%     vec = [repmat(v,1,nS),zeros(1,nS)];
% 
% end