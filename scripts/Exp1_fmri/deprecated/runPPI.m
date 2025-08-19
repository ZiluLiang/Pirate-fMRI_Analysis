function runPPI(SPMmat_dir,voi_name,ppiname)
    [directory,fmri,filepattern]  = get_pirate_defaults(false,'directory','fmri','filepattern');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PSYCHO-PHYSIOLOGIC INTERACTION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % GENERATE PPI STRUCTURE
    %==========================================================================
    matlabbatch{1}.spm.stats.ppi.spmmat = cellstr(fullfile(SPMmat_dir,'SPM.mat'));
    matlabbatch{1}.spm.stats.ppi.type.ppi.voi = cellstr(fullfile(SPMmat_dir,strcat(voi_name,'.mat')));
    matlabbatch{1}.spm.stats.ppi.type.ppi.u = [2 1 -1; 3 1 1];
    matlabbatch{1}.spm.stats.ppi.name = ppiname;
    matlabbatch{1}.spm.stats.ppi.disp = 0;
    
    % OUTPUT DIRECTORY
    %==========================================================================
    matlabbatch{2}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(SPMmat_dir);
    matlabbatch{2}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'PPI';
    
    % MODEL SPECIFICATION
    %==========================================================================
    
    % Directory
    %--------------------------------------------------------------------------
    matlabbatch{3}.spm.stats.fmri_spec.dir = cellstr(fullfile(SPMmat_dir,'PPI'));
    
    % Timing
    %--------------------------------------------------------------------------
    matlabbatch{3}.spm.stats.fmri_spec.timing.units = 'scans';
    matlabbatch{3}.spm.stats.fmri_spec.timing.RT = fmri.tr;
    
    % Session
    %--------------------------------------------------------------------------
    f = spm_select('FPList', fullfile(SPMmat_dir,'functional'), '^snf.*\.img$');
    matlabbatch{3}.spm.stats.fmri_spec.sess.scans = cellstr(f);
    
    % Regressors
    %--------------------------------------------------------------------------
    matlabbatch{3}.spm.stats.fmri_spec.sess.multi_reg = {...
        fullfile(data_path,'GLM','PPI_V2x(Att-NoAtt).mat');...
        fullfile(data_path,'multi_block_regressors.mat')};
    
    % High-pass filter
    %--------------------------------------------------------------------------
    matlabbatch{3}.spm.stats.fmri_spec.sess.hpf = 192;
    
    % MODEL ESTIMATION
    %==========================================================================
    matlabbatch{4}.spm.stats.fmri_est.spmmat = cellstr(fullfile(data_path,'PPI','SPM.mat'));
    
    % INFERENCE
    %==========================================================================
    matlabbatch{5}.spm.stats.con.spmmat = cellstr(fullfile(data_path,'PPI','SPM.mat'));
    matlabbatch{5}.spm.stats.con.consess{1}.tcon.name = 'PPI-Interaction';
    matlabbatch{5}.spm.stats.con.consess{1}.tcon.weights = [1 0 0 0 0 0 0];
    
    % RESULTS
    %==========================================================================
    matlabbatch{6}.spm.stats.results.spmmat = cellstr(fullfile(data_path,'PPI','SPM.mat'));
    matlabbatch{6}.spm.stats.results.conspec.contrasts = 1;
    matlabbatch{6}.spm.stats.results.conspec.threshdesc = 'none';
    matlabbatch{6}.spm.stats.results.conspec.thresh = 0.01;
    matlabbatch{6}.spm.stats.results.conspec.extent = 3;
    matlabbatch{6}.spm.stats.results.print = false;
    
    spm_jobman('run',matlabbatch);
    
    % JUMP TO V5 AND OVERLAY ON A STRUCTURAL IMAGE
    %--------------------------------------------------------------------------
    spm_mip_ui('SetCoords',[39 -72 0]);
    spm_sections(xSPM,findobj(spm_figure('FindWin','Interactive'),'Tag','hReg'),...
        fullfile(data_path,'structural','nsM00587_0002.img'));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PSYCHO-PHYSIOLOGIC INTERACTION GRAPH
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    clear matlabbatch
    
    % VOI: EXTRACTING TIME SERIES: V5 [39 -72 0]
    %==========================================================================
    matlabbatch{1}.spm.util.voi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
    matlabbatch{1}.spm.util.voi.adjust = 1;
    matlabbatch{1}.spm.util.voi.session = 1;
    matlabbatch{1}.spm.util.voi.name = 'V5';
    matlabbatch{1}.spm.util.voi.roi{1}.spm.spmmat = {''};
    matlabbatch{1}.spm.util.voi.roi{1}.spm.contrast = 3;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.threshdesc = 'FWE';
    matlabbatch{1}.spm.util.voi.roi{1}.spm.thresh = 0.001;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.extent = 3;
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.centre = [39 -72 0];
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.radius = 6;
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.move.local.spm = 1;
    matlabbatch{1}.spm.util.voi.expression = 'i1 & i2';
    
    % GENERATE PPI STRUCTURE: V2xNoAtt
    %==========================================================================
    matlabbatch{2}.spm.stats.ppi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
    matlabbatch{2}.spm.stats.ppi.type.ppi.voi = cellstr(fullfile(data_path,'GLM','VOI_V2_1.mat'));
    matlabbatch{2}.spm.stats.ppi.type.ppi.u = [2 1 1];
    matlabbatch{2}.spm.stats.ppi.name = 'V2xNoAtt';
    matlabbatch{2}.spm.stats.ppi.disp = 0;
    
    % GENERATE PPI STRUCTURE: V2xAtt
    %==========================================================================
    matlabbatch{3}.spm.stats.ppi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
    matlabbatch{3}.spm.stats.ppi.type.ppi.voi = cellstr(fullfile(data_path,'GLM','VOI_V2_1.mat'));
    matlabbatch{3}.spm.stats.ppi.type.ppi.u = [3 1 1];
    matlabbatch{3}.spm.stats.ppi.name = 'V2xAtt';
    matlabbatch{3}.spm.stats.ppi.disp = 0;
    
    % GENERATE PPI STRUCTURE: V5xNoAtt
    %==========================================================================
    matlabbatch{4}.spm.stats.ppi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
    matlabbatch{4}.spm.stats.ppi.type.ppi.voi = cellstr(fullfile(data_path,'GLM','VOI_V5_1.mat'));
    matlabbatch{4}.spm.stats.ppi.type.ppi.u = [2 1 1];
    matlabbatch{4}.spm.stats.ppi.name = 'V5xNoAtt';
    matlabbatch{4}.spm.stats.ppi.disp = 0;
    
    % GENERATE PPI STRUCTURE: V5xAtt
    %==========================================================================
    matlabbatch{5}.spm.stats.ppi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
    matlabbatch{5}.spm.stats.ppi.type.ppi.voi = cellstr(fullfile(data_path,'GLM','VOI_V5_1.mat'));
    matlabbatch{5}.spm.stats.ppi.type.ppi.u = [3 1 1];
    matlabbatch{5}.spm.stats.ppi.name = 'V5xAtt';
    matlabbatch{5}.spm.stats.ppi.disp = 0;
    
    spm_jobman('run',matlabbatch);
end
