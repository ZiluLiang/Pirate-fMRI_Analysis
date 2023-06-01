[directory,participants]  = get_pirate_defaults(false,'directory','participants');
glm_name = 'sc_navigation';
glm_dir  = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\sc_navigation';
glm_config = get_glm_config(glm_name); 

clear colin rcond_res
rcond_res = nan(participants.nvalidsub,2);
R = cell(participants.nvalidsub,5);
colin = cell(participants.nvalidsub,1);
trs = 296;
for isub = 1:participants.nvalidsub
    subSPM = load(fullfile(glm_dir,'first',participants.validids{isub},'SPM.mat'),'SPM').SPM;
    designMatrix = subSPM.xX.xKXs.X;%spm_filter(subSPM.xX.K, subSPM.xX.W*subSPM.xX.X);
    designMatrixR = subSPM.xX.X;
    
    rcond_res(isub,1) = rcond(designMatrixR*designMatrixR');
    rcond_res(isub,2) = rcond(designMatrix*designMatrix');
    rcond_res(isub,3) = rcond(full(subSPM.xX.W));
    rcond_res(isub,4) = rcond(full(subSPM.xVi.V));
    rcond_res(isub,5) = rcond(full(subSPM.xVi.Cy));
%     for k = 1%:4
%         sessrow = (k-1)*trs+1:k*trs;
%         taskreg_pat = {sprintf('Sn(%d) stimuli',k),sprintf('Sn(%d) response',k)};
%         hemoreg_pat = {sprintf('Sn(%d) R',k)};
%         taskreg_idx = find_regressor_idx(subSPM,taskreg_pat);
%         hemoreg_idx = find_regressor_idx(subSPM,hemoreg_pat);
%         taskreg   = designMatrix(sessrow,taskreg_idx);
%         hemoreg   = designMatrix(sessrow,hemoreg_idx);
%         R{isub,k} = corr(taskreg,hemoreg);
%         tmpr(k) = mean(abs(R{isub,k}),'all');
%         sessX   = designMatrix(:,[taskreg_idx,hemoreg_idx]);
%         ncol    = size(sessX,2);
%         colin{isub}(:,k) = arrayfun(@(j) fitlm(sessX(:,1:ncol~=j),sessX(:,j)).Rsquared.Adjusted,1:ncol);
%     end
%     R{isub,5} = corr(taskreg);
%     colin{isub}(:,5) = mean(colin{isub}(:,1:end),2);
%    rcond_res(isub,5) = mean(tmpr,'all');
%    rcond_res(isub,6) = mean(colin{isub}(1:2,5),'all');
%    rcond_res(isub,7) = mean(colin{isub}(3:end,5),'all');
end
rcond_res(:,6) = 1:30';
rcond_res_T = array2table(rcond_res,'VariableNames',{'rcond_X','rcond_WX','rcond_W','rcond_V','rcond_Cy','subid'});
