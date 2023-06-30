function check_spatial_registration(comparetype,subimg_dir,varargin)
% shortcut to display different images side by side to check spatial registration quality
% shortcuts for three types of comparisons are specified:
%     1) all2template: compare all modality to template
%     2) anat2epi: compare t1 to epi
%     3) epi2template: compare epi to template
% see documentions in each embedded functions for details on what images
% are selected for display
% example usage: check_spatial_registration(comparetype,subimg_dir)
% INPUTS:
%     - comparetype: type of comparison, must be one of the above types
%     - subimg_dir: directory that contains the rp file and preprocessed
%     image of participants
% -----------------------------------------------------------------------    
% Author: Zilu Liang
     
    % find images to display and configure display settings
    config_handles = struct('all2template',@compare_all2template,...
                            'anat2epi',    @compare_anat2epi,...
                            'epi2template',@compare_epi2template);
    [filepattern,directory] = get_pirate_defaults(false,'filepattern','directory');
    % find templates
    if isfile(directory.mni_template)
        template = directory.mni_template;
    else
        template = fullfile(spm('Dir'),'canonical','avg152T1.nii');
    end
    
    [captions,images,overlays] = config_handles.(comparetype)(subimg_dir,filepattern,template,varargin{:});
    view_cfg = cell2struct([captions;images;overlays],{'captions','images','overlays'});
    
    % display image
    ext_spm_checkreg(view_cfg);
end

%% --------------- handy functions for setting up viewing configurations for frequently used spatial registration checks --------------
function [captions,images,overlays] = compare_all2template(subimg_dir,filepattern,template,varargin)
    % Before/After reorientation:
    % show all the acquired imgs and the the template image side by side 
    % to check if manual reorientation and manually setting origin is neccessary.     
    
    % display settings
    if numel(varargin) == 1 && islogical(varargin{1})
        flag_display_pre_reorientaion = varargin{1};
    else
        flag_display_pre_reorientaion = false;
    end

    % get reoriented images
    if ~flag_display_pre_reorientaion
        nii_files   = structfun(@(scantype) ...
                                  structfun(@(pattern) spm_select('FPList', subimg_dir, [pattern,'.*\.nii']),...
                                             scantype,'UniformOutput',false),...
                               filepattern.reorient,...
                               'UniformOutput',false);
    end
    
    % if do not want to display reoriented images or reoriented imgaes are
    % not found, get the raw images.
    if ~flag_display_pre_reorientaion && all(structfun(@(scantype) all(structfun(@(scans) ~isempty(scans),scantype)),nii_files))
        flag_display_pre_reorientaion = false;
        nii_files = structfun(@(scantype) structfun(@(scans) cellstr(scans),scantype,'uni',0),nii_files,'uni',0);
    else
        flag_display_pre_reorientaion = true;
        nii_files  = structfun(@(scantype) ...
                                    structfun(@(pattern) ...
                                                  cellstr(spm_select('FPList', subimg_dir, [pattern,'.*\.nii'])),...
                                              scantype,'UniformOutput',false),...
                               filepattern.raw,'UniformOutput',false);
    end    
    
    % select the first volume of the first functional runs of each task to display
    func_imgs  = struct2cell(structfun(@(taskscan) cellstr(strrep(taskscan{1},'.nii','.nii,1')),nii_files.functional,'uni',0));
    func_imgs  = cat(1,func_imgs{:})';
    %fmap_imgs  = struct2cell(nii_files.fieldmap);
    %fmap_imgs  = cat(1,fmap_imgs{:})';
    
    images   = [{template},nii_files.anatomical.T1,func_imgs,nii_files.fieldmap.phasediff];
    overlays = repmat({''},size(images));
    captions = [{'template(MNI152 gen6 nlin sym)'}...
                {'T1'},...
                cellfun(@(x) [x,'run1 vol1'],fieldnames(nii_files.functional),'uni',0)',...
                {'fieldmap - phasediff'}];
            
    prefix   = {'reoriented','raw'};
    captions(2:end) = cellfun(@(x) [prefix{flag_display_pre_reorientaion+1},' ',x],captions(2:end),'uni',0);
end


function [captions,images,overlays] = compare_anat2epi(subimg_dir,filepattern,template,varargin)
    % Before/After coreg:
    % show the mean epi overlaid on the t1 image to check the quality of corregistration.
    
    % display settings
    if numel(varargin) == 1 && islogical(varargin{1})
        flag_display_pre_coreg = varargin{1};
    else
        flag_display_pre_coreg = false;
    end
    
    meanepi  = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.meanepi,'.*.nii']));
    tpm_gm   = {[spm('dir'),filesep,'tpm',filesep,'TPM.nii,1']};
    coreg_t1 = spm_select('FPList',subimg_dir,[filepattern.preprocess.coreg,'.*.nii']);
    
    %If reorientation is performed before preprocessing, diplay reoriented images to compare
    [~,meanepi_fn,~] = fileparts(meanepi);
    startind = regexp(meanepi_fn,strrep(filepattern.reorient.functional.task1,'^',''), 'once');
    if isempty(startind)
        t1          = cellstr(spm_select('FPList',subimg_dir,[filepattern.raw.anatomical.T1,'.nii']));
    else
        t1          = cellstr(spm_select('FPList',subimg_dir,[filepattern.reorient.anatomical.T1,'.nii']));
    end    
    
    if ~flag_display_pre_coreg && ~isempty(coreg_t1)
        images   = [{coreg_t1},coreg_t1,tpm_gm,template];
        overlays = [meanepi,{'','',''}];
        captions = {'meanepi overlaid on coregistered T1','coregistered T1','TPM - Grey matter','template(MNI152 gen6 nlin sym)'};
    else
        images   = [meanepi,t1,tpm_gm,template];
        overlays = {'','','',''};
        captions = {'meanepi','T1','TPM - Grey matter','template(MNI152 gen6 nlin sym)'};
    end
    
end

function [captions,images,overlays] = compare_epi2template(subimg_dir,filepattern,template,varargin)
    % show the mean/first normalized epi, normalized segmented anatomical images, 
    % TPM for GM and WM, and the mni-152 template side by side 
    % as well as the mean normalized epi overlaid on the mni-152 
    % to check the quality of normalization and segmentation.

    if numel(varargin) == 1 && islogical(varargin{1})
        flag_usemeanepi = varargin{1};
    else
        flag_usemeanepi = false;
    end
    
    if flag_usemeanepi
        %find mean functional image or create if it doesn't exist
        taskepi  = spm_select('FPList',subimg_dir,[filepattern.preprocess.meanepi_wu,'.*.nii']);
        if isempty(taskepi)
            fprintf('calculating mean epi for all normalized task fmri images')
            taskepi = create_mean_normalizedepi(subimg_dir,filepattern);
        end
        taskcap  = 'mean task epi';
    else
        taskepi  = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.normalise,'.*',strrep(filepattern.raw.functional.task1,'^',''),'.*.nii']));
        taskepi  = [taskepi{1},',1'];
        taskcap  = 'first task epi';
    end

    anat_gm = spm_select('FPList',subimg_dir,[filepattern.preprocess.normseg_t1,'1.*.nii']);
    anat_wm = spm_select('FPList',subimg_dir,[filepattern.preprocess.normseg_t1,'2.*.nii']);

    tpm_gm  = {[spm('dir'),filesep,'tpm',filesep,'TPM.nii,1']};
    tpm_wm  = {[spm('dir'),filesep,'tpm',filesep,'TPM.nii,2']};

    images   = [taskepi,template,tpm_wm,anat_wm,anat_gm,tpm_gm];
    captions = {taskcap,'template(MNI152 gen6 nlin sym)'...
                'sub T1 GM','TPM GM',...
                'sub T1 WM','TPM WM',};
    overlays = {'','','','','',''};
end


function meantaskepi_filename = create_mean_normalizedepi(subimg_dir,filepattern)
    normalized_epis =  cellstr(spm_select('List',subimg_dir,[filepattern.preprocess.normalise,'.*.nii']));    
    normalized_epi_vols = cellfun(@(x) cellstr(spm_select('ExtFPList',subimg_dir,x)),normalized_epis,'uni',0);
    
    calmeanepi = {};
    calmeanepi{1}.spm.util.imcalc.input = cat(1,normalized_epi_vols{:});
    calmeanepi{1}.spm.util.imcalc.output = 'qc_meanwuepi';
    calmeanepi{1}.spm.util.imcalc.outdir = {subimg_dir};
    calmeanepi{1}.spm.util.imcalc.expression = 'mean(X)';
    calmeanepi{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    calmeanepi{1}.spm.util.imcalc.options.dmtx = 1;
    calmeanepi{1}.spm.util.imcalc.options.mask = 0;
    calmeanepi{1}.spm.util.imcalc.options.interp = 1;
    calmeanepi{1}.spm.util.imcalc.options.dtype = 4;
    
    spm_jobman ('run',calmeanepi);

    meantaskepi_filename = fullfile(calmeanepi{1}.spm.util.imcalc.outdir,[calmeanepi{1}.spm.util.imcalc.output,'.nii']);
end