clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');

glm_dir = fullfile(directory.fmri_data, "unsmoothedLSA", "LSA_stimuli_navigation_movdir1col2pmod");

subid_list = participants.validids;

DesignMatrix = struct();
for i = 1:numel(subid_list)
    firstlvl_dir = fullfile(glm_dir,"first",subid_list{i});
    subSPM = load(fullfile(firstlvl_dir,"SPM.mat")).SPM;
    DesignMatrix.xX{i} = subSPM.xX.X; % this is the convolved design matrix
    DesignMatrix.xKXs{i} = subSPM.xX.xKXs.X; % this is the filtered and whitened
    DesignMatrix.name{i} = cellfun(@(x) regexprep(x,'(?i)sn\((\d+)\)\s*([^*]*)(?:\*bf\(\d+\))?', 's$1_$2'), subSPM.xX.name,'uni',0);
    DesignMatrix.subid{i} = subid_list{i};
    DesignMatrix.sessSEIDX{i} = [1,cumsum(subSPM.nscan)];
    %corr(wf_desMat)

    % Now let's get the contrast from our SPM file
    % xCon=horzcat(subSPM.xCon(:).c)';
    % xCon=xCon(:,1:3);
end
save('TreasureHunt_DesignMatrix_movdir1col2pmod.mat', '-struct', 'DesignMatrix', '-v7.3');

dmat = subSPM.xX.X;
%dmat(dmat==0) = nan;
h = heatmap(dmat);
h.MissingDataColor = [1 1 1]; % white
h.MissingDataLabel = '';
h.ColorLimits = [min(dmat(:)) max(dmat(:))];

% Nc = length(CM);
% bCM = {};
% for c = 1:Nc
%     bCM{c} = kron(CM{c},eye(Nk)/Nk);
% end
% 
% iXX = pinv(X'*X);
% for c = 1:Nc
% 	e(1,c) = 1/trace(bCM{c}*iXX*bCM{c}');
%     fprintf('Contrast %d: Efficiency = %f\n',c,e(1,c))
% end
% fprintf('\n')