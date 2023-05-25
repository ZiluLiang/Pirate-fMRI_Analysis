function mean_tsnr = calculate_snr(source,outputname,masks)
% This script calculates temporal signal to noise ratio
% The script will generate a nii image with tsnr for each voxel
% if masks is specified, mean tsnr will be calculated within each masked
% regions and returned; if masks is not specified, will return the mean
% tsnr across all voxels.
% usage: calculate_snr(source,outputname,masks)
% INPUT: 
%    -source: the full path of the 4D nii series tsnr to be calculated
%    -outputname: the full path of the output tsnr image
%    -masks: a cell array of mask image files or a struct with fieldnames as
%    mask names and field values as mask image files.
% ------ written by Zilu Liang(2023.5,Oxford)------


    if nargin<2
        [filepath,src_name,ext] = fileparts(source);
        outputname = fullfile(filepath,['tsnr_',src_name,ext]); 
    end
    
    if nargin<3, masks = {}; end
    if iscell(masks) && ~isempty(masks)
        masknames = arrayfun(@(x) sprintf('mask%d',x),1:numel(masks),'uni',0);
        masks = cell2struct(reshape(masks,numel(masks),1),masknames);
    end
    if isempty(masks)
        masks = cell2struct({fullfile(spm('dir'),'tpm','mask_ICV.nii')},{'spmICV'});
    end

    disp('Extracting Time Series...')
    Vi = spm_vol(source);
    % get 4d timeseries, the first three dimension are spatial coordinates the fourth is time
    ts_4D = spm_read_vols(Vi,1); 

    %transform into 2D to speed up computation
    disp('Calculating tSNR...')
    [Nx,Ny,Nz,Nt] = size(ts_4D);
    ts_2D = reshape(ts_4D,Nx*Ny*Nz,Nt);
    %[Nv,Nt] = size(ts_2D);
    mean_acrosstime = mean(ts_2D,2);
    std_acrosstime = std(ts_2D,0,2);
    tsnr_2D = mean_acrosstime./ std_acrosstime;
    tsnr_3D = reshape(tsnr_2D,Nx,Ny,Nz);

    Vo = Vi(1);
    Vo.fname = outputname;
    spm_write_vol(Vo,tsnr_3D);
    
    % return snr measure within masks
    mean_tsnr = structfun(@(x) mean(spm_summarise(Vo,x),'all','omitnan'),masks,'uni',0);
end