function o = ext_spm_coregcost(source_img,template_img,varargin)
% o = ext_spm_coregcost(source_img,template_img,costfun) 
% -- adapted from spm_coreg.m according to https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1812&L=SPM&P=R16843
% this function is a simplified version of part of the code from spm_coreg.m
% it returns the match between source image and template imgage based on 
% the cost function specified. 
% INPUT:
%    - source_img:cell array containing one/multiple images to be compared
%    - template_image: one template image
%    - costfun: must be one of: 'mi','ecc','nmi','ncc'.
% OUTPUT:
%    - 1*N array of matching measures, N being the number of source images.
% TODO: mi measures are greater than one, is that normal?

    if numel(varargin) == 1 && ischar(varargin{1})
        costfun = varargin{1};
    else
        costfun = 'nmi';
    end
    VG = spm_vol(template_img);
    VF = spm_vol(source_img);
    o = nan(size(VF));
    if ~isfield(VG, 'uint8')
        VG.uint8 = loaduint8(VG);
        vxg      = sqrt(sum(VG.mat(1:3,1:3).^2));
        fwhmg    = sqrt(max([1 1 1]*2^2 - vxg.^2, [0 0 0]))./vxg;
        VG       = smooth_uint8(VG,fwhmg); % Note side effects
    end
    for k=1:numel(VF)
        if size(VF{k},1)>1
            warning('4D data found, using the first volume of the time series!')
            VFk = VF{k}(1);
        else
            VFk = VF{k};
        end
        if ~isfield(VFk, 'uint8')
            VFk.uint8 = loaduint8(VFk);
            vxf       = sqrt(sum(VFk.mat(1:3,1:3).^2));
            fwhmf     = sqrt(max([1 1 1]*2^2 - vxf.^2, [0 0 0]))./vxf;
            VFk       = smooth_uint8(VFk,fwhmf); % Note side effects
        end
        o(k) = -cost(VG,VFk,costfun);
    end
end
%==========================================================================
% function o = cost(VG,VF,cf,fwhm) -- stolen from optfun in spm_coreg.m
%==========================================================================

function o = cost(VG,VF,cf,fwhm)
    % The function that is minimised.
    if nargin<4, fwhm = [7 7];   end
    if nargin<3, cf   = 'mi';    end



    % Create the joint histogram
    H = spm_hist2(VG.uint8,VF.uint8, eye(4) ,[1 1 1]);

    % Compute cost function from histogram
    H  = H+eps;
    sh = sum(H(:));
    H  = H/sh;
    s1 = sum(H,1);
    s2 = sum(H,2);

    switch lower(cf)
        case 'mi'
            % Mutual Information:
            H   = H.*log2(H./(s2*s1));
            mi  = sum(H(:));
            o   = -mi;
        case 'ecc'
            % Entropy Correlation Coefficient of:
            % Maes, Collignon, Vandermeulen, Marchal & Suetens (1997).
            % "Multimodality image registration by maximisation of mutual
            % information". IEEE Transactions on Medical Imaging 16(2):187-198
            H   = H.*log2(H./(s2*s1));
            mi  = sum(H(:));
            ecc = -2*mi/(sum(s1.*log2(s1))+sum(s2.*log2(s2)));
            o   = -ecc;
        case 'nmi'
            % Normalised Mutual Information of:
            % Studholme,  Hill & Hawkes (1998).
            % "A normalized entropy measure of 3-D medical image alignment".
            % in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
            nmi = (sum(s1.*log2(s1))+sum(s2.*log2(s2)))/sum(sum(H.*log2(H)));
            o   = -nmi;
        case 'ncc'
            % Normalised Cross Correlation
            i     = 1:size(H,1);
            j     = 1:size(H,2);
            m1    = sum(s2.*i');
            m2    = sum(s1.*j);
            sig1  = sqrt(sum(s2.*(i'-m1).^2));
            sig2  = sqrt(sum(s1.*(j -m2).^2));
            [i,j] = ndgrid(i-m1,j-m2);
            ncc   = sum(sum(H.*i.*j))/(sig1*sig2);
            o     = -ncc;
        otherwise
            error('Invalid cost function specified');
    end
end
%==========================================================================
% function udat = loaduint8(V) -- stolen from spm_coreg.m
%==========================================================================
function udat = loaduint8(V)
    % Load data from file indicated by V into an array of unsigned bytes.
    if size(V.pinfo,2)==1 && V.pinfo(1) == 2
        mx = 255*V.pinfo(1) + V.pinfo(2);
        mn = V.pinfo(2);
    else
        spm_progress_bar('Init',V.dim(3),...
            ['Computing max/min of ' spm_file(V.fname,'filename')],...
            'Planes complete');
        mx = -Inf; mn =  Inf;
        for p=1:V.dim(3)
            img = spm_slice_vol(V,spm_matrix([0 0 p]),V.dim(1:2),1);
            img = img(isfinite(img));
            mx  = max([max(img(:))+paccuracy(V,p) mx]);
            mn  = min([min(img(:)) mn]);
            spm_progress_bar('Set',p);
        end
    end

    % Another pass to find a maximum that allows a few hot-spots in the data.
    spm_progress_bar('Init',V.dim(3),...
            ['2nd pass max/min of ' spm_file(V.fname,'filename')],...
            'Planes complete');
    nh = 2048;
    h  = zeros(nh,1);
    for p=1:V.dim(3)
        img = spm_slice_vol(V,spm_matrix([0 0 p]),V.dim(1:2),1);
        img = img(isfinite(img));
        img = round((img+((mx-mn)/(nh-1)-mn))*((nh-1)/(mx-mn)));
        h   = h + accumarray(img,1,[nh 1]);
        spm_progress_bar('Set',p);
    end
    tmp = [find(cumsum(h)/sum(h)>0.9999); nh];
    mx  = (mn*nh-mx+tmp(1)*(mx-mn))/(nh-1);

    % Load data from file indicated by V into an array of unsigned bytes.
    spm_progress_bar('Init',V.dim(3),...
        ['Loading ' spm_file(V.fname,'filename')],...
        'Planes loaded');
    udat = zeros(V.dim,'uint8');
    st = rand('state'); % st = rng;
    rand('state',100); % rng(100,'v5uniform'); % rng('defaults');
    for p=1:V.dim(3)
        img = spm_slice_vol(V,spm_matrix([0 0 p]),V.dim(1:2),1);
        acc = paccuracy(V,p);
        if acc==0
            udat(:,:,p) = uint8(max(min(round((img-mn)*(255/(mx-mn))),255),0));
        else
            % Add random numbers before rounding to reduce aliasing artifact
            r = rand(size(img))*acc;
            udat(:,:,p) = uint8(max(min(round((img+r-mn)*(255/(mx-mn))),255),0));
        end
        spm_progress_bar('Set',p);
    end
    spm_progress_bar('Clear');
    rand('state',st); % rng(st);
end
%==========================================================================
% function acc = paccuracy(V,p)
%==========================================================================
function acc = paccuracy(V,p)
    if ~spm_type(V.dt(1),'intt')
        acc = 0;
    else
        if size(V.pinfo,2)==1
            acc = abs(V.pinfo(1,1));
        else
            acc = abs(V.pinfo(1,p));
        end
    end
end
%==========================================================================
% function V = smooth_uint8(V,fwhm)
%==========================================================================
function V = smooth_uint8(V,fwhm)
    % Convolve the volume in memory (fwhm in voxels).
    lim = ceil(2*fwhm);
    x  = -lim(1):lim(1); x = spm_smoothkern(fwhm(1),x); x  = x/sum(x);
    y  = -lim(2):lim(2); y = spm_smoothkern(fwhm(2),y); y  = y/sum(y);
    z  = -lim(3):lim(3); z = spm_smoothkern(fwhm(3),z); z  = z/sum(z);
    i  = (length(x) - 1)/2;
    j  = (length(y) - 1)/2;
    k  = (length(z) - 1)/2;
    spm_conv_vol(V.uint8,V.uint8,x,y,z,-[i j k]);
end