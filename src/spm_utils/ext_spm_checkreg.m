function ext_spm_checkreg(varargin)
% customised script based on spm_check_registration and spm_orthviews
% display list of images side by side, with overlays or captions if specfied
% usage:
%    ext_spm_checkreg(images,captions,overlays) where all inputs are
%    cells. captions and overlays can be omitted.
% or 
%    ext_spm_checkreg(view_cfg) where view_cfg is a struct with the
%    following fields:'images','captions','overlays'
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    % initialize empty configurations the same size as the 
    init_cfg = struct('images',[],'captions',[],'overlays',[]);
    cfg_fields = {'images','captions','overlays'};

    % check if inputs are valid
    error_flag = true;
    if numel(varargin) == 1 && isstruct(varargin{1}) && all(ismember(fieldnames(varargin{1}),fieldnames(init_cfg)))
        view_cfg = varargin{1};
        error_flag = false;
    end
    
    if numel(varargin) >= 1 && numel(varargin) <= 3 && all(cellfun(@(arg) iscell(arg),varargin))
        ns = cellfun(@(arg) numel(arg),varargin);
        if all(ns == ns(1))
            view_cfg = cell2struct(cat(1,varargin{:}),cfg_fields(1:numel(varargin)));
            error_flag = false;
        end
    end

    % combine inputs with default configurations
    if error_flag
        error('invalid inputs')
    else
        for j = 1:numel(cfg_fields) - numel(fieldnames(view_cfg))
            missing_field = cfg_fields{end-(j-1)};
            [view_cfg.(missing_field)] = deal(init_cfg.(missing_field));
        end
    end
     
    % display images
    spm_check_registration(view_cfg.images);
    for i = 1:size(view_cfg,1)
        if ~isempty(view_cfg(i).captions)
            spm_orthviews('Caption',i,view_cfg(i).captions);
        end
        if ~isempty(view_cfg(i).overlays)
            spm_orthviews('AddColouredImage',i,view_cfg(i).overlays,[1 0 0]);
        end
    end
    spm_orthviews('MaxBB')
    spm_orthviews('Reposition',[0,0,0])
end