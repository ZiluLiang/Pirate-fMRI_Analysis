function nuisance_reg = generate_nuisance_regressor(subimg_dir,flag_save,terms,saving_dir)
    %valid terms:{'raw','fw','squared-raw','squared-fw','outlier-fw'}
    
    %configurations
    if nargin<2, flag_save = true; end
    if nargin<3, terms = {'raw','fw'}; end% terms to be included as nuisance regressor, by default will generate 12, i.e., realignment parameters and their first derivatives(framewise displacement)
    if nargin<4, saving_dir = subimg_dir; end    
    
    % get the filepattern from default setting 
    filepattern = get_pirate_defaults(false,'filepattern');
    
    % find files 
    rp_files   = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.motionparam,'.*.txt']));
    
    % setup headmotion regressors
    n_sessions = numel(rp_files);
    tasknames = cell(n_sessions,1);
    for j = 1:n_sessions
        [~,tasknames{j},~] = fileparts(rp_files{j});
        tasknames{j} = regexprep(tasknames{j},filepattern.preprocess.motionparam,'');
        tasknames{j} = regexprep(tasknames{j},filepattern.preprocess.reorient,'');        
        scanningparam = loadjson(spm_select('FPList',subimg_dir,[tasknames{j},'.json']));
        
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
        OFW = zeros(nvol);
        FW_mm = FW;
        FW_mm(:,4:6) = FW_mm(:,4:6).*50;
        FW_mm = arrayfun(@(x) sum(abs(FW_mm(x,:))),1:size(FW_mm));
        OFW(FW_mm>scanningparam.SliceThickness,FW_mm>scanningparam.SliceThickness) = 1;
        OFW = OFW.* eye(nvol);        
        
        % set up nuisance regressors
        nuisance_reg = [];
        if ismember('raw',terms), nuisance_reg = [nuisance_reg,D]; end %#ok<*AGROW>
        if ismember('fw',terms),  nuisance_reg = [nuisance_reg,FW]; end
        if ismember('squared-raw',terms), nuisance_reg = [nuisance_reg,D2]; end
        if ismember('squared-fw',terms),  nuisance_reg = [nuisance_reg,FW2]; end
        if ismember('outlier-fw',terms),  nuisance_reg = [nuisance_reg,OFW]; end
        
        if flag_save
            checkdir(saving_dir)
            prefix = strrep(filepattern.preprocess.nuisance,'^','');
            writematrix(nuisance_reg,fullfile(saving_dir,[prefix,tasknames{j},'.txt']),'Delimiter','tab');            
        end
    end
end
