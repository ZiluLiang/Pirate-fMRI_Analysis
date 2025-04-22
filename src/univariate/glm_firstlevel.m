function glm_firstlevel(nii_files,multicond_files,multireg_files,output_dir)
% Specify first-level glm in spm
% usage: glm_firstlevel(nii_files,multicond_files,nuisance_files,output_dir,flag_estimate)
% INPUT:
% - nii_files: cell array of fmri time series files for the analysis, each
%               element is the file for one session
% - multicond_files: cell array of .mat file that contains the specification of
%                    conditions in spm, each element is the file for one session.  
%                    see spm batch GUI for detailed formatting instructions
% - multireg_files:  cell array of .txt/.mat files that contains the 
%                    multiple regressors not convolved with hrf, e.g.
%                    nuisance regressors, head motion regressors etc
%                    each element is the file for one session.  
%                    each column in a txt file or in the R matrix of the .mat
%                    file is the value of one nuisance regressor. 
% - output_dir:      the directory in which result images and SPM.mat is saved 
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    if numel(nii_files)==numel(multicond_files) && numel(nii_files)==numel(multireg_files)
        nsess = numel(nii_files);
    else
        error('number of 4D nii files, multiple conditions files, and number of nuisance regressor files do not match!')
    end   
    checkdir(output_dir)

    
    %% Directory
    specification.dir = {output_dir};
    
    %% Timing parameters
    % check out discussion here on how to specify timing parameters for
    % multiband data https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind2205&L=SPM&P=R5597
    specification.timing.units   = 'secs'; %timing mode, options in clude {'scans','secs'}
    specification.timing.RT      = 1.73;   %TR length
    specification.timing.fmri_t  = 50/2;   %number of time bins: number of slices/multi band factor = 50/2
    specification.timing.fmri_t0 = 13;     %set the middle time bins as reference
    
    %% Data&Design
    for iSess = 1:nsess
        %Data&Design - scans/imgs to be processed
        specification.sess(iSess).scans = nii_files{iSess};%#ok<*AGROW> %scans for each run
        %Data&Design - conditions and pmods
        specification.sess(iSess).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
        specification.sess(iSess).multi = multicond_files{iSess};
        %Data&Design - nuisance regressors
        specification.sess(iSess).regress = struct('name',{},'val',{});
        specification.sess(iSess).multi_reg = multireg_files{iSess};
        % high-pass filter
        specification.sess(iSess).hpf = 180;%changed on 2024 02 24
    end
    
    %% Factorial Design
    specification.fact = struct('name',{},'levels',{});
    
    %% Basis Functions:
    %use the default hrf,first element means time derivatives, second means
    %dispersion derivatives
    specification.bases.hrf.derivs = [0 0];
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
    % implicit masking threshold. SPM by default uses an implicit masking
    % threshold of 0.8. But this sometimes exclude voxels with low tsnr in
    % regions we care about. Here we use a combination of explicit mask
    % with low implicit masking threshold so that sensible voxels in all
    % brain regions can be included. This threshold is purely empirical
    % without theoretical support. For example discussion see: https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1910&L=SPM&P=R91828
    specification.mthresh = 0.2; 
    
    %% Explicit mask 
    %%%use spm intracranial volume mask as explicit mask
    specification.mask = {fullfile(spm('dir'),'tpm','mask_ICV.nii')};
    %%specification.mask = {''};
    
    %% Serial correlations
    specification.cvi='AR(1)';%spm default, or change to 'none' if do not want to account for temporal autocorrelation;
    
    %% Save batchjob and run
    matlabbatch{1}.spm.stats.fmri_spec = specification;
    save(fullfile(specification.dir{1},'model_specification.mat'),'matlabbatch')
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);
end