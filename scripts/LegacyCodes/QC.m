function QC(check_field,varargin)
    qc_handles  = struct('distortion_correction', @check_distortion_correction,...
                         'coregistration',        @(varargin) check_spatial_registration('anat2epi',varargin),...
                         'normalization',         @(varargin) check_spatial_registration('epi2template',varargin),...
                         'motion_correction',     @check_headmotion);
                     
    if ismember(check_field,fieldnames(qc_handles))
        qc_handles.(check_field)(varargin{:})
    else
        error('invalid check measure! must be one of the following:\n %s.\n',strjoin(fieldnames(qc_handles),', '))
    end
end