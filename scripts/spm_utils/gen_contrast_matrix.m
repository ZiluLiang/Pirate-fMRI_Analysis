function w_mat = gen_contrast_matrix(subSPM,wm)
% generate contrast vector
% INPUTS:
%  - subSPM: path to first level SPM.mat file or the loaded SPM struct;
%  - wm: contrast matrix in one session
% TODO: generate contrasts based on regressor names?
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
    nS = numel(subSPM.nscan);
    nuisance_idx = find_regressor_idx(subSPM,'Sn(1) R');
    nN = numel(nuisance_idx);
    w_mat = contrast_matrix(wm,nN,nS);
end

function w_mat = contrast_matrix(wm,nN,nS)
% generate contrast vector
% based on spm convention that regressors are ordered as in:
% [C-1_sess-1,C-2_sess-1,...N-1_sess-1,N-2_sess-1,...C-nC_sess-N,...N-nN_sess-nS,S-1,...S-nS]
% where C is condition regressor, N is nuisance regressor and S is the session regressor
% INPUTS:
% wm  - contrast matrix in one session
% nN  - number of nuisance regressors
% nS  - number of sessions
%    wm        = wm./nS;
    nC        = size(wm,1);
    sess_wmat = [wm,zeros(nC,nN)];
    w_mat     = [repmat(sess_wmat,1,nS),zeros(nC,nS)];
end
