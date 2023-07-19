clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');

par_dirs = {'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\localizer_no_selection',...
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\concatall_no_selection'};
for x = 1:2
    searchlight_rsa_dir = par_dirs{x};
    analysis = {'correlation','cosinesimilarity','regression'};       
    for j_analysis = 2%numel(analysis)
        rsa_dir = fullfile(searchlight_rsa_dir,analysis{j_analysis});

        switch analysis{j_analysis}
            case "correlation"
                metric_names = fieldnames(loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);
                img_regexp ="rho_%04d.nii";
                get_new_imgname = @(x) strcat('transformed_',x);                
                trans_formula = 'atanh(i1)';
            case "regression"
                metric_names = fieldnames(loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);
                img_regexp ="beta_%04d.nii";
                get_new_imgname = @(x) x;
                trans_formula = '';
            case "cosinesimilarity"
                metric_names = loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.resultnames;
                img_regexp ="ps_%04d.nii";                
        end

        for k = 1:numel(metric_names)
            cd(fullfile(directory.projectdir,'scripts'))
            curr_metric = metric_names{k};
            
            if analysis{j_analysis} == "cosinesimilarity"
                if contains(curr_metric,'within')
                    get_new_imgname = @(x) strcat('transformed_',x);
                    trans_formula = 'i1-1/9';
                else
                    get_new_imgname = @(x) x;
                    trans_formula = '';
                end
            end

            img_transform = @(firstlvldir,x) spm_imcalc(char(fullfile(firstlvldir,x)),char(fullfile(firstlvldir,get_new_imgname(x))),trans_formula,struct("dtype",16));        

            outputdir = fullfile(rsa_dir,'second',curr_metric);
            if exist(outputdir,'dir')
                disp('exist')
                rmdir(outputdir,'s')                        
            end
            
            metric_imgname = sprintf(img_regexp,k-1);
            
            
            if ~isempty(trans_formula)
                cellfun(@(subid) img_transform(fullfile(rsa_dir,'first',subid),metric_imgname),participants.validids);
                scans = cellstr(fullfile(rsa_dir,'first',participants.validids,get_new_imgname(metric_imgname)));                
            else
                scans = cellstr(fullfile(rsa_dir,'first',participants.validids,metric_imgname));
            end

            glm_grouplevel(outputdir,{scans},{curr_metric});
            glm_estimate(outputdir)
            glm_contrast(outputdir,cellstr(curr_metric),{[1]});
            glm_results(outputdir,1,struct('type','none','val',0.001,'extent',0),{'xls'})
        end
    end
end