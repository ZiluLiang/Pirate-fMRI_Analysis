function checkdir(varargin)
% usage: checkdir(dir1,dir2,...,dirn)
% usage: checkdir({dir1,dir2,...,dirn})
% usage: checkdir({dir1,dir2},{dir3,...,dirn)
% check if directories exist, if not will create the directories
% -----------------------------------------------------------------------    
% Author: Zilu Liang

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