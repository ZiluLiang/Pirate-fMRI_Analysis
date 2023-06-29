function glm_results(SPMmat_dir,ths_cfg,output_fmt,masks)
    spm('defaults','FMRI')
    
    default_threshold = struct('type','none',...
                               'val',0.001,...
                               'extent',0);
    nC = numel(load(fullfile(SPMmat_dir,'SPM.mat')).SPM.xCon);
    
    try 
        if isstruct(ths_cfg)
            if numel(ths_cfg) ~= nC, ths_cfg = repmat(ths_cfg(1),nC); end
        else
            ths_cfg = repmat(default_threshold,nC);
        end
    catch 
        ths_cfg = repmat(default_threshold,nC);
    end
    
    if nargin<3
        output_fmt  = {'jpg','xls'};
    else
        valid_fmt = {'ps','eps','pdf','jpg','png','tif','fig','csv','xls'};
        output_fmt = output_fmt(cellfun(@(fmt) contains(fmt,valid_fmt),output_fmt));
    end
    
    if nargin<4, apply_mask  = false; end
	
    results.spmmat(1) = cellstr(fullfile(SPMmat_dir,'SPM.mat'));
    for k = 1:nC
        results.conspec(k).titlestr = '';
        results.conspec(k).contrasts = k;
        results.conspec(k).threshdesc = ths_cfg(k).type;
        results.conspec(k).thresh = ths_cfg(k).val;
        results.conspec(k).extent = ths_cfg(k).extent;
        results.conspec(k).conjunction = 1;
        if ~apply_mask
            results.conspec(k).mask.none = 1;
        else
            results.conspec.mask.image.name = masks{k};
            results.conspec.mask.image.mtype = 0;
        end
    end
    results.units=1;
    for j = 1:numel(output_fmt)
        results.export{j}.(output_fmt{j})=true;
    end
    matlabbatch{1}.spm.stats.results = results;
    spm_jobman('run', matlabbatch);
end