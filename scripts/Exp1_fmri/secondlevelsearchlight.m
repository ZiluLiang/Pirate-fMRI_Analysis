% compare correlation coefficient or regression coefficient to zero
%% No Interaction
%clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');

par_dirs = {
            fullfile(directory.fmri_data,'unsmoothedLSA','rsa_searchlight','fourruns_noselection_mvnn_averageall'),...            
            fullfile(directory.fmri_data,'unsmoothedLSA','rsa_searchlight','localizer_noselection_mvnn_averageall'),...
            fullfile(directory.fmri_data,'unsmoothedLSA','rsa_searchlight','compositionRetrieval_noselection_mvnn_atoe')...
            };
analysis = {'regression','correlation','composition'};   
run_rec = cell(numel(par_dirs),numel(analysis));

x = 3;
searchlight_rsa_dir = par_dirs{x};        

j_analysis = 3;
rsa_folders = {dir(fullfile(searchlight_rsa_dir,analysis{j_analysis})).name};
rsa_folders = rsa_folders(cellfun(@(x) ~contains(x,'.'),rsa_folders));
rsa_dirs = fullfile(searchlight_rsa_dir,analysis{j_analysis},rsa_folders);
run_rec{x}{j_analysis} = cell(numel(rsa_dirs),1);

for j_rsadir = 1:numel(rsa_dirs)
    rsa_dir = rsa_dirs{j_rsadir};
    switch analysis{j_analysis}
        case "correlation"
            if contains(rsa_dir,'betweenrun_rdm_corr')
                metric_names = cellstr('betweenrun_rdm_corr');
            else
                metric_names = fieldnames(loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);
            end
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
        case "composition"
            metric_names = loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.resultnames;
            metric_idxs = 1:numel(metric_names);
            img_regexp ="beta_%04d.nii";
            get_new_imgname = @(x) x;
            trans_formula = '';
    end
    run_rec{x}{j_analysis}{j_rsadir} = cell(numel(metric_names),1);
    for k = 1:numel(metric_names)
        %sprintf("x=%d, j_analysis=%d, j_rsadir=%d, k=%d",x,j_analysis,j_rsadir,k)
        cd(fullfile(directory.projectdir,'scripts'))
        curr_metric = metric_names{k};
        metric_imgname = sprintf(img_regexp,metric_idxs(k)-1);
    
        outputdir = fullfile(rsa_dir,'second_nosmooth',curr_metric);
        if exist(outputdir,'dir')
            disp('remove existing directory')
            rmdir(outputdir,'s')                        
        end

        if ~isempty(trans_formula)
            img_transform = @(firstlvldir,x) spm_imcalc(char(fullfile(firstlvldir,x)),char(fullfile(firstlvldir,get_new_imgname(x))),trans_formula,struct("dtype",16));
            cellfun(@(subid) img_transform(fullfile(rsa_dir,'first',subid),metric_imgname),participants.validids);
        end

        %smooth first level images
        all_sub_dirs = fullfile(rsa_dir,'first',participants.validids);
        cellfun(@(subimg_dir) smooth(subimg_dir,get_new_imgname(metric_imgname)), all_sub_dirs);

        perform second level analysis
        scans = cellstr(fullfile(rsa_dir,'first',participants.validids,strcat("s",get_new_imgname(metric_imgname) )));
        scans_g = cellstr(fullfile(rsa_dir,'first',participants.generalizerids,strcat("s",get_new_imgname(metric_imgname) )));
        scans_ng = cellstr(fullfile(rsa_dir,'first',participants.nongeneralizerids,strcat("s",get_new_imgname(metric_imgname) )));
        
        try
            glm_grouplevel(fullfile(outputdir,'allparticipants'),'t1',{scans},{curr_metric});
            glm_estimate(fullfile(outputdir,'allparticipants'));
            glm_contrast(fullfile(outputdir,'allparticipants'),cellstr(curr_metric),{[1]});
            
            glm_grouplevel(fullfile(outputdir,'g_vs_ng'),'t2',{scans_g,scans_ng},{curr_metric});
            glm_estimate(fullfile(outputdir,'g_vs_ng'));
            glm_contrast(fullfile(outputdir,'g_vs_ng'), ...
                        {strcat('G>NG: ',curr_metric),strcat('G:',curr_metric),strcat('NG:',curr_metric)}, ...
                        {[1,-1],[1,0],[0,1]});
            %glm_results(fullfile(outputdir,'generalizer_vs_nongeneralizer'),1,struct('type','none','val',0.001,'extent',0),{'csv'});
            
            glm_grouplevel(fullfile(outputdir,'generalizer_only'),'t1',{scans_g},{curr_metric});
            glm_estimate(fullfile(outputdir,'generalizer_only'));
            glm_contrast(fullfile(outputdir,'generalizer_only'),cellstr(curr_metric),{[1]});

            run_rec{x}{j_analysis}{j_rsadir}{k} = 1;
        catch error
            run_rec{x}{j_analysis}{j_rsadir}{k} = error;
        end
        % 
    end
end
