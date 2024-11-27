function [nuisance_regs,nN] = generate_nuisance_regressor(subimg_dir,flag_save,terms,saving_dir)
% This function generates the nuisance regressor for first level GLM
% analysis from the rp_*.txt files generated in the spm realignment
% process. 
% INPUT:
%    - subimg_dir: the directory where rp_*.txt files are saved
%    - flag_save:  write the generated regressor to file or not
%    - terms:      terms to be included as nuisance regressor, must be from
%                  {'raw','fw','squared-raw','squared-fw','outlier-fw'}.
%                  'raw': the six columns in the rp_*.txt file
%                  'fw':  the six columns of framewise displacement, i.e.,
%                  the first derivatives of the six columns in the rp_*.txt file.
%                  'outlier-fw': the outliers in framewise displacement
%                  using the voxel size as a threshold. rotation parameters
%                  in radian were transformed into mm on a 50-mm sphere
%                  (proximately the radius of adult human head) before
%                  classification of outliers.
%    - saving_dir: the directory where the nuisance regressor files are
%    saved
% OUTPUT:
%    - nuisance_regs: the matrices of nuisance regressors stored in struct
%    - nN: number of nuisance regressors
% -----------------------------------------------------------------------    
% Author: Zilu Liang

   
    % get the filepattern from default setting 
    [filepattern,fmri] = get_pirate_defaults(false,'filepattern','fmri');

    %configurations
    if nargin<2, flag_save  = true; end
    % terms to be included as nuisance regressor, by default will generate 12, i.e., realignment parameters and their first derivatives(framewise displacement)
    if nargin<3, terms      = fmri.nuisance_terms; end
    if nargin<4, saving_dir = subimg_dir; end    
    
    
    % find files
    rp_files   = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.motionparam,'.*.txt']));
    
    % setup headmotion regressors
    n_sessions = numel(rp_files);
    tasknames = cell(n_sessions,1);
    nuisance_regs = struct();
    for j = 1:n_sessions
        [~,tasknames{j},~] = fileparts(rp_files{j});
        tasknames{j} = regexprep(tasknames{j},filepattern.preprocess.motionparam,'');
        tasknames{j} = regexprep(tasknames{j},filepattern.preprocess.reorient,'');        
        
        % displacement(D)
        D = table2array(readtable(rp_files{j})); 
        
        % framewise displacement(FW)
        FW = D - [zeros(1,6);D(1:end-1,:)];
        
        % get squared term
        D2 = D.*D;
        FW2 = FW.*FW;
        
        % get outliers - create nvol*nvol matrix OFW, for volume j with
        % fw>voxelsize, set OFW(j,j) = 1
        nvol = size(D,1);
        FW_mm = FW;
        FW_mm(:,4:6) = FW_mm(:,4:6).*50;
        FW_mm = arrayfun(@(x) max(abs(FW_mm(x,:))),1:size(FW_mm)); % should use sum or max instead?
        
        OFW0 = zeros(nvol);
        OFW0(sub2ind(size(OFW0), find(FW_mm>fmri.voxelsize), find(FW_mm>fmri.voxelsize))) = 1;
        OFW0 = OFW0(:,~all(OFW0==0,1)); % volumes when excessive hm occur
        OFW_pre1  = [OFW0(2:end,:);zeros(1,size(OFW0,2))]; % 1 volume before ehm occur
        OFW_post1 = [zeros(1,size(OFW0,2));OFW0(1:end-1,:)]; % 1 volume after ehm occur
        OFW_post2 = [zeros(2,size(OFW0,2));OFW0(1:end-2,:)]; % 2nd volume after ehm occur
        
        OFWtmp = [OFW0,OFW_pre1,OFW_post1,OFW_post2];
        OFW = unique(OFWtmp.','rows').';

        % set up nuisance regressors
        nuisance_reg = [];
        if ismember('raw',terms), nuisance_reg = [nuisance_reg,D]; end %#ok<*AGROW>
        if ismember('fw',terms),  nuisance_reg = [nuisance_reg,FW]; end
        if ismember('squared-raw',terms), nuisance_reg = [nuisance_reg,D2]; end
        if ismember('squared-fw',terms),  nuisance_reg = [nuisance_reg,FW2]; end
        if ismember('outlier-fw',terms),  nuisance_reg = [nuisance_reg,OFW]; end
        
        nuisance_regs.(strrep(regexprep(tasknames{j},'sub-.*_task-',''),'_run-','')) = nuisance_reg;
        %disp(size(OFW))
        %disp(size(nuisance_reg))
        if flag_save
            checkdir(saving_dir)
            prefix = strrep(filepattern.preprocess.nuisance,'^','');
            writematrix(nuisance_reg,fullfile(saving_dir,[prefix,tasknames{j},'.txt']),'Delimiter','tab');            
        end
        nN = size(nuisance_reg,2);
    end
end
