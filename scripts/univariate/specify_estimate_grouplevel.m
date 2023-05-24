function specify_estimate_grouplevel(varargin)
% specify_estimate_grouplevel(model_dir,contrast_names)
    err_flag = 1;
    if nargin == 2 && ischar(varargin{1}) && iscell(varargin{2})
        model_dir      = varargin{1};
        contrast_names = varargin{2};
        err_flag = 0;
    end
    if err_flag
        error('invalid inputs')
    else
        n_contrast       = numel(contrast_names);
        seclvl_contrasts = 1:n_contrast;        
        seclvl_dirs      = fullfile(model_dir,'second',contrast_names);
    end
    
    participants  = get_pirate_defaults(false,'participants');
        
    for j=seclvl_contrasts   
        %% Directory
        checkdir(seclvl_dirs{j})
        design.dir = seclvl_dirs(j);
        %% Design
        %%select files for second level analysis
        design.des.t1.scans = fullfile(model_dir,'first',participants.validids,sprintf('con_000%d.nii',j));
        %%covariate
        design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
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
        
        %% Contrast Specification
        contrast.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        contrast.consess{1}.tcon.name = contrast_names{j};
        contrast.consess{1}.tcon.sessrep = 'none';
        contrast.consess{1}.tcon.weights = 1;
        contrast.delete=0;
        
        %% Output Results
        results.spmmat(1) = cfg_dep('Contrast Manager: SPM.mat File', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        for k = 1:numel(contrast.consess)
            results.conspec(k).titlestr = '';
            results.conspec(k).contrasts = k;
            results.conspec(k).threshdesc = 'none';
            results.conspec(k).thresh = 0.001;
            results.conspec(k).extent = 0;
            results.conspec(k).conjunction = 1;
            results.conspec(k).mask.none = 1;
        end
        results.units=1;
        results.export{1}.jpg=true;
        results.export{2}.xls=true;        
        
        %% Save second level batch job
        matlabbatch{1}.spm.stats.factorial_design = design;
        matlabbatch{2}.spm.stats.fmri_est = estimation;
        matlabbatch{3}.spm.stats.con = contrast;
        matlabbatch{4}.spm.stats.results = results;
        save(fullfile(seclvl_dirs{j},'spec2est2ndlvl.mat'),'matlabbatch')
        
        spm_jobman('run', matlabbatch);
    end
end
