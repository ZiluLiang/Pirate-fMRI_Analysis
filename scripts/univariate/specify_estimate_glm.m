function specify_estimate_glm(varargin)
%writen by Zilu Liang (2023 May, Oxford)
%this script specifies first-level glm in spm
% usage: specify_estimate_glm(nii_files,multicond_files,nuisance_files,output_dir,flag_estimate)

    % validate input and find files 
    
    err_flag = 1;
    if nargin == 4 || nargin == 5
        if all(cellfun(@(arg) iscell(arg),varargin(1:3))) && ischar(varargin{4})
            nii_files       = varargin{1};
            multicond_files = varargin{2};
            nuisance_files  = varargin{3};
            output_dir      = varargin{4};
            err_flag   = 0;
            if nargin<5, flag_estimate = true; end
        end
    end
    
    if err_flag
        error('invalid inputs')
    else
        checkdir(output_dir)
    end
    
    if ~all(cellfun(@(x) exist(x,'file'),[nii_files,multicond_files,nuisance_files]))
        all_files = [nii_files,multicond_files,nuisance_files];
        missing_files = all_files(cellfun(@(x) ~exist(x,'file'),[nii_files,multicond_files,nuisance_files]));
        error('The following files do not exist: \n %s \n',strjoin(missing_files,'\n'))
    end
    if numel(nii_files)==numel(multicond_files) && numel(nii_files)==numel(nuisance_files)
        nsess = numel(nii_files);
    else
        error('number of 4D nii files, multiple conditions files, and number of nuisance regressor files do not match!')
    end   
    
    
    %% Directory
    specification.dir = {output_dir};
    
    %% Timing parameters
    % check out discussion here on how to specify timing parameters for
    % multiband data https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind2205&L=SPM&P=R5597
    specification.timing.units   = 'secs'; %timing mode, options in clude {'scans','secs'}
    specification.timing.RT      = 1.73;   %TR length
    specification.timing.fmri_t  = 50/2;   %number of slices/multi band factor = 50/2
    specification.timing.fmri_t0 = 13;     %reference slice
    
    %% Data&Design
    for iSess = 1:nsess
        %Data&Design - scans/imgs to be processed
        specification.sess(iSess).scans = nii_files(iSess);%#ok<*AGROW> %scans for each run
        %Data&Design - conditions and pmods
        specification.sess(iSess).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
        specification.sess(iSess).multi = multicond_files(iSess);
        %Data&Design - nuisance regressors
        specification.sess(iSess).regress = struct('name',{},'val',{});
        specification.sess(iSess).multi_reg = nuisance_files(iSess);
        % high-pass filter
        specification.sess(iSess).hpf = 128;
    end
    
    %% Factorial Design
    specification.fact = struct('name',{},'levels',{});
    
    %% Basis Functions:
    %use the default hrf,first element means time derivatives, second means
    %dispersion derivatives
    %specification.bases.hrf.derivs = [0 0];
    % %to use other basis function, specify through
    % %1)fourier set
    % specification.fourier.length =;
    % specification.fourier.order =;
    % %2)fourier set (hanning)
    % specification.fourier_han.length =;
    % specification.fourier_han.order =;
    % %3)gamma functions
    % specification.gamma.length =;
    % specification.gamma.order =;
    % %4)finite impulse sequence
    % specification.fir.length =;
    % specification.fir.order =;
    % %5)no basis functions
    % specification.none = true;

    %% Model Interactions(Volterra)
    %1=do not model interactions 2=model interactions
    specification.volt = 1;
    
    %% Global Normalisation
    specification.global = 'None';% or 'Scaling'
    
    %% Masking Threshold
    specification.mthresh = -Inf;
    
    %% Explicit mask 
    %%%spm intracranial volume mask: {'D:\Matlab_Toolbox\spm12\tpm\mask_ICV.nii'}; no mask: {''}
    specification.mask = {fullfile(spm('dir'),'tpm','mask_ICV.nii')};
    %%specification.mask = {''};
    
    %% Serial correlations
    specification.cvi='AR(1)';%'FAST','none';
    
    %% Estimation
    if flag_estimate
        estimation.spmmat(1)        = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        estimation.write_residuals  = 0;
        estimation.method.Classical = 1;        
    end
    %% Save batchjob and run
    matlabbatch{1}.spm.stats.fmri_spec = specification;
    if flag_estimate
        matlabbatch{2}.spm.stats.fmri_est = estimation;
    end
    save(fullfile(specification.dir{1},'spec2est1stlvl.mat'),'matlabbatch')
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end