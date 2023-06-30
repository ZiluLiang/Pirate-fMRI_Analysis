function glm_results(SPMmat_dir,contrasts,ths_cfg,output_fmt,masks)
% report of results of contrasts specified in spm.mat file
% INPUT:
% - SPMmat_dir: directory to spmMAT file
% - ths_cfg: threshold settings for computing results
% - output_fmt: output format. default is {'jpg','xls'}
% - masks: mask to be applied when computing results
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    spm('defaults','FMRI')
    xCon = load(fullfile(SPMmat_dir,'SPM.mat')).SPM.xCon;
    nC = numel(load(fullfile(SPMmat_dir,'SPM.mat')).SPM.xCon);
    def_contrast_idxs = 1:nC;
    if nargin<2, contrasts = def_contrast_idxs;end    
    if isstring(contrasts) || ischar(contrasts),contrasts = {contrasts}; end
    if isnumeric(contrasts) && any(arrayfun(@(x) ismember(x,def_contrast_idxs),contrasts))
        contrast_idxs = contrasts(arrayfun(@(x) ismember(x,def_contrast_idxs),contrasts));
    elseif iscell(contrasts) && any(cellfun(@(x) ismember(x,{xCon.name}),contrasts))
        search_cnames = contrasts(cellfun(@(x) ismember(x,{xCon.name}),contrasts));
        [contrast_idxs,~,~] = find_contrast_idx(fullfile(SPMmat_dir,'SPM.mat'),search_cnames);
    else
        error('cannot find contrast with name %s\n', strjoin(contrasts,', '))
    end

    default_threshold = struct('type','none',...
                               'val',0.001,...
                               'extent',0);
    
    
    try 
        if isstruct(ths_cfg)
            if numel(ths_cfg) ~= nC, ths_cfg = repmat(ths_cfg(1),nC); end
        else
            ths_cfg = repmat(default_threshold,nC);
        end
    catch 
        ths_cfg = repmat(default_threshold,nC);
    end
    
    if nargin<4
        output_fmt  = {'jpg','xls'};
    else
        valid_fmt = {'ps','eps','pdf','jpg','png','tif','fig','csv','xls'};
        output_fmt = output_fmt(cellfun(@(fmt) contains(fmt,valid_fmt),output_fmt));
    end
    
    if nargin<5, apply_mask  = false; end
	
    results.spmmat(1) = cellstr(fullfile(SPMmat_dir,'SPM.mat'));
    for k = 1:numel(contrast_idxs)
        results.conspec(k).titlestr = '';
        results.conspec(k).contrasts = contrast_idxs(k);
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