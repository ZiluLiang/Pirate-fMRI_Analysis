function meanepi_filename = create_mean_epi(subimg_dir,filepattern,varargin)
%  usage create_mean_epi(subimg_dir,filepattern,outputname,outdir)
% INPUT: 
%  - subimg_dir: the directory to the epi images
%  - filepattern: regular expression to search for epi images to calculate
%  mean from
%  - outputname: output name of the mean epi image, if not specified, will
%  be named as meanepi_filepattern.nii
%  - outdir: directory where output mean epi image will be saved. If not
%  specifyed, output mean epi image will be saved to subimg_dir
%
% This function will create:
%   outputname.nii
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    
    if numel(varargin) >= 1
        outputname = varargin{1};
    else
        illegal_strs = {'^', '.', '*', '[', ']'};
        suffix = filepattern;
        for j = 1:numel(illegal_strs)
            suffix = strrep(suffix,'^','');
        end
        outputname = ['meanepi_',suffix];
    end
    
    if numel(varargin) >= 2
        outdir = varargin(2);
    else
        outdir = {subimg_dir};
    end

    normalized_epis     =  cellstr(spm_select('FPList',subimg_dir,filepattern));    
    normalized_epi_vols = cellfun(@(x) cellstr(spm_select('expand',x)),normalized_epis,'uni',0);
    n_vols = numel(normalized_epi_vols);
    expression = ['(',sprintf('i%d+',1:n_vols-1),sprintf('i%d)/%d',n_vols,n_vols)];    
    
    calmeanepi = {};
    calmeanepi{1}.spm.util.imcalc.input = cat(1,normalized_epi_vols{:});
    calmeanepi{1}.spm.util.imcalc.output = outputname;
    calmeanepi{1}.spm.util.imcalc.outdir = outdir;
    calmeanepi{1}.spm.util.imcalc.expression = expression;
    calmeanepi{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    calmeanepi{1}.spm.util.imcalc.options.dmtx = 0;
    calmeanepi{1}.spm.util.imcalc.options.mask = 0;
    calmeanepi{1}.spm.util.imcalc.options.interp = 1;
    calmeanepi{1}.spm.util.imcalc.options.dtype = 4;
    
    spm_jobman ('run',calmeanepi);

    meanepi_filename = fullfile(calmeanepi{1}.spm.util.imcalc.outdir,[calmeanepi{1}.spm.util.imcalc.output,'.nii']);
end