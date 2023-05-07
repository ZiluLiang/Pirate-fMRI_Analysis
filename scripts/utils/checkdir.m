function checkdir(varargin)
dirs=varargin(cellfun(@(x) ~iscell(x),varargin));
cellfun(@(x) checksingledir(x),dirs)

dircell=varargin(cellfun(@(x) iscell(x),varargin));
cellfun(@(x) cellfun(@(y) checksingledir(y),x),dircell)

function checksingledir(x)
    if ~exist(x,'dir')
        mkdir(x); 
    end
end
end