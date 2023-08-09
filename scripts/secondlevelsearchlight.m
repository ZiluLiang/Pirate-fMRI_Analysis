clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');

par_dirs = {
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\localizer_no_selection',...
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\concatall_no_selection',...
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight\localizer_no_selection',...
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight\concatall_no_selection'
            };
pardir = "D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight";
par_dirs = fullfile(pardir,{dir(fullfile(pardir,'*selection')).name});

par_dirs = {
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\concatall_no_selection',...
            ...%'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\oddeven_no_selection',...
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\noconcatall_no_selection',...
            'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\unsmoothedLSA\rsa_searchlight\fourruns_no_selection',...
            %'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight\concatall_no_selection',...
            %'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight\oddeven_no_selection',...
            %'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight\noconcatall_no_selection',...
            %'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\smoothed5mmLSA\rsa_searchlight\fourruns_no_selection',...
            };

parfor x = 1:numel(par_dirs)
    searchlight_rsa_dir = par_dirs{x};
    analysis = {'decoding_AxisLocContinuous','decoding_AxisLocDiscrete','correlation','cosinesimilarity','regression'};       
    for j_analysis = 3%1:numel(analysis)
        rsa_dir = fullfile(searchlight_rsa_dir,analysis{j_analysis});

        switch analysis{j_analysis}
            case "correlation"
                metric_names = fieldnames(loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);
                metric_idxs = 1:numel(metric_names);
                img_regexp ="rho_%04d.nii";
                get_new_imgname = @(x) strcat('transformed_',x);                
                trans_formula = 'atanh(i1)';
            case "regression"
                metric_names = fieldnames(loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);
                metric_idxs = 1:numel(metric_names);
                img_regexp ="beta_%04d.nii";
                get_new_imgname = @(x) x;
                trans_formula = '';
            case "cosinesimilarity"
                metric_names = loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.resultnames;
                metric_idxs = 1:numel(metric_names);
                img_regexp ="ps_%04d.nii";    
            case "decoding_AxisLocDiscrete"
                allmetric_names = loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.resultnames;
                metric_names = allmetric_names(cellfun(@(x) contains(x,'evaluation'),allmetric_names));
                metric_idxs = cellfun(@(x) find(strcmp(allmetric_names,x)),metric_names);
                img_regexp ="acc_%04d.nii";    
                get_new_imgname = @(x) strcat('transformed_',x);                
                trans_formula = 'i1-1/5';
            case "decoding_AxisLocContinuous"
                allmetric_names = loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.resultnames;
                metric_names = allmetric_names(cellfun(@(x) contains(x,'evaluation'),allmetric_names));
                metric_idxs = cellfun(@(x) find(strcmp(allmetric_names,x)),metric_names);
                img_regexp ="rsquare_%04d.nii";    
                get_new_imgname = @(x) x;
                trans_formula = '';
        end

        if contains(analysis{j_analysis},'decoding')
            allmetric_names = loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.resultnames;
            training_performance_names = allmetric_names(cellfun(@(x) contains(x,'training'),allmetric_names));
            training_performance_idxs  = cellfun(@(x) find(strcmp(allmetric_names,x)),training_performance_names);
            for k = 1:numel(training_performance_names)
                curr_t = training_performance_names{k};
                training_performance_imgname = sprintf(img_regexp,training_performance_idxs(k)-1);
                outputdir = fullfile(rsa_dir,'second',curr_t);
                checkdir(outputdir)
                training_performance_scans = fullfile(rsa_dir,'first',participants.validids,training_performance_imgname);
                spm_imcalc(char(training_performance_scans),...
                           char(fullfile(outputdir,strcat(curr_t,'_meanacc.nii'))),...
                           'mean(X)',struct("dtype",16,"dmtx",1));
            end
        end

        for k = 1:numel(metric_names)
            cd(fullfile(directory.projectdir,'scripts'))
            curr_metric = metric_names{k};

            if analysis{j_analysis} == "cosinesimilarity"
                if contains(curr_metric,'within')
                    get_new_imgname = @(x) strcat('transformed_',x);
                    if contains(curr_metric,'test')
                        trans_formula = 'i1-2/15';
                    else
                        trans_formula = 'i1-1/9';
                    end
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

            metric_imgname = sprintf(img_regexp,metric_idxs(k)-1);


            if ~isempty(trans_formula)
                cellfun(@(subid) img_transform(fullfile(rsa_dir,'first',subid),metric_imgname),participants.validids);
            end
            scans = cellstr(fullfile(rsa_dir,'first',participants.validids,get_new_imgname(metric_imgname)));
            scans_g = cellstr(fullfile(rsa_dir,'first',participants.generalizerids,get_new_imgname(metric_imgname)));
            scans_ng = cellstr(fullfile(rsa_dir,'first',participants.nongeneralizerids,get_new_imgname(metric_imgname)));

            glm_grouplevel(fullfile(outputdir,'allparticipants'),'t1',{scans},{curr_metric});
            glm_estimate(fullfile(outputdir,'allparticipants'))
            glm_contrast(fullfile(outputdir,'allparticipants'),cellstr(curr_metric),{[1]});
            glm_results(fullfile(outputdir,'allparticipants'),1,struct('type','none','val',0.001,'extent',0),{'xls'})

            glm_grouplevel(fullfile(outputdir,'generalizer_vs_nongeneralizer'),'t2',{scans_g,scans_ng},{curr_metric});
            glm_estimate(fullfile(outputdir,'generalizer_vs_nongeneralizer'))
            glm_contrast(fullfile(outputdir,'generalizer_vs_nongeneralizer'), ...
                        {strcat('G>NG: ',curr_metric),strcat('G:',curr_metric),strcat('NG:',curr_metric)}, ...
                        {[1,-1],[1,0],[0,1]});
            glm_results(fullfile(outputdir,'generalizer_vs_nongeneralizer'),1,struct('type','none','val',0.001,'extent',0),{'xls'})

        end
    end
end