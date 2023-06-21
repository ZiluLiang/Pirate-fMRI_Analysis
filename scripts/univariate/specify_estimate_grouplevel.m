function specify_estimate_grouplevel(outputdir,scans,factor_names,cov)
% specify_estimate_grouplevel(outputdir,scans)
% INPUT:
%  - otputdir: output directory of second level analysis
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

    if nargin<4, cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {}); end
        
    %% Directory
    checkdir(outputdir)
    design.dir = {outputdir};

    %% Design
    if nF == 1 && nLs(1) ==1
        % if only one cell, it is a one sample t test
        t1 = struct();
        t1.scans = reshape(scans{1},[],1);
        % assign to design struct
        design.des.t1 = t1;
    else
        fd = struct();
        for f = 1:nF
            fd.fact(f).name = factor_names{f};
            fd.fact(f).levels = nLs(f);
            fd.fact(f).dept = 0;
            fd.fact(f).variance = 1;
            fd.fact(f).gmsca = 0;
            fd.fact(f).ancova = 0;
        end
        for j = 1:numel(scans)
            fd.icell(j).levels = ind2sub(nLs,j);
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

    %% Estimation
    estimation.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    estimation.write_residuals = 0;
    estimation.method.Classical = 1;

    %% Save second level batch job
    matlabbatch{1}.spm.stats.factorial_design = design;
    matlabbatch{2}.spm.stats.fmri_est = estimation;
    save(fullfile(outputdir,'spec2est2ndlvl.mat'),'matlabbatch')

    spm_jobman('run', matlabbatch);
end
