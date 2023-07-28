function glm_grouplevel(outputdir,design_type,scans,factor_names,within_factors,cov)
% glm_grouplevel(outputdir,scans,factor_names,cov)
% INPUT:
%  - otputdir: output directory of second level analysis
%  - design_type: type of second level analysis, can be
%               (1) t1: one-sample t-test
%               (2) t2: two-sample t-test
%               (3) factorial: fullfactorial design
%  - scans: a N1*N2*...*Nf...*NnF cell array, it specifies a design with nF
%           number of factors, factor 1 has N1 levels, factor 2 has N2
%           levels, etc, and the nF th factor has NnF number of levels.
%           each element contains the first-level contrast images corresponding 
%           to one condition in the factorial design. For example, a design
%           has 4 factors A,B,C,D, yielding a total of nA*nB*nC*nD
%           conditions. Then scans must be the saize of nA*nB*nC*nD, and
%           scans{1,2,3,4} specifies the first level contrasts images 
%           corresponding to condition A1B2C3D4.
%  - factor_names: cell array of names of each factor. 
%  - within_factors: logical array of whether each factor is a within-partcipant 
%  - cov: covariates
% -----------------------------------------------------------------------    
% Author: Zilu Liang

% TODO: need to support multiple covariates, extend to other types of
% second level analysis
    
    valid_design_types = {'t1','t2','factorial'};
    if ~ismember(design_type,valid_design_types)
        error("invalid design type for second level glm")
    end

    % reshape input so that each factor at least has two levels
    nLs0 = size(scans); 
    nLs  = nLs0(nLs0>1); % number of levels for each factor
    if numel(nLs)==0 
        factor_names = factor_names(1);
        nLs = 1;
    elseif numel(nLs)==1
        scans = reshape(scans,[nLs,1]);
        factor_names = factor_names(1);
    else
        scans = reshape(scans,nLs);
        factor_names = factor_names(nLs0>1);
    end
    % number of factors
    nF  = numel(factor_names);
    
    if nargin<5, within_factors = zeros(size(factor_names)); end
    if nargin<6, cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {}); end
        
    %% Directory
    checkdir(outputdir)
    design.dir = cellstr(outputdir);

    %% Design
    if nF == 1 && nLs(1) ==1, design_type="t1"; end % if only one cell, it is a one sample t test
    switch design_type
        case "t1"
            t1 = struct();
            t1.scans = reshape(scans{1},[],1);
            % assign to design struct
            design.des.t1 = t1;
        case "t2"            
            t2.scans1 = scans{1};
            t2.scans2 = scans{2};
            t2.dept = 0;
            t2.variance = 1;
            t2.gmsca = 0;
            t2.ancova = 0;
            design.des.t2 = t2;
        case "factorial"
            fd = struct();
            for f = 1:nF
                fd.fact(f).name = factor_names{f};
                fd.fact(f).levels = nLs(f);
                fd.fact(f).dept = within_factors(f);
                fd.fact(f).variance = 1;
                fd.fact(f).gmsca = 0;
                fd.fact(f).ancova = 0;
            end
            for j = 1:numel(scans)
                [a,b] = ind2sub(nLs,j);
                if nF==1
                    fd.icell(j).levels = a;
                else
                    fd.icell(j).levels = [a,b];
                end
                fd.icell(j).scans  = reshape(scans{j},[],1);
            end
            fd.contrasts = 1;
            % assign to design struct
            design.des.fd = fd;
    end
    
    %%covariate
    design.cov = cov;
    design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    %%masking and other settings
    design.masking.tm.tm_none = 1;
    design.masking.im = 1;
    design.masking.em = {''};
    design.globalc.g_omit = 1;
    design.globalm.gmsca.gmsca_no = 1;
    design.globalm.glonorm = 1;

    %% Save second level batch job
    matlabbatch{1}.spm.stats.factorial_design = design;
    save(fullfile(outputdir,'specification2ndlvl.mat'),'matlabbatch')

    spm_jobman('run', matlabbatch);
end
