% compare correlation coefficient of different model RDM


% clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
par_dirs = {
            fullfile(directory.fmri_data,'unsmoothedLSA','rsa_searchlight','fourruns_noselection_mvnn_averageall'),...            
            fullfile(directory.fmri_data,'unsmoothedLSA','rsa_searchlight','localizer_noselection_mvnn_aoe')
            };

%% Compare PTA vs Euc in Generalizers in training stimuli pairs

m1rsa_dir = fullfile(par_dirs{1},"correlation","PTA");
m1_metric_names = fieldnames(loadjson(fullfile(m1rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);
m1_idx = find(strcmp(metric_names,"trainstimpairs_gtlocEuclidean"));
arrayfun(@(idx) char(fullfile(rdm_corr_coef_dir, sprintf("stransformed_rho_%04d.nii",idx-1))), compare_metrics_indices, 'uni', 0 );

subsets = {"between", "within"};
for j = 1:numel(subsets)
    comp_name = sprintf("%s_teststimpairs_featvsloc",subsets{j});
    outputdir = fullfile(rsa_dir,"compare_rdm_corrcoefs",comp_name);
    firstlvl_dir = fullfile(outputdir,'first');
    secondlvl_dir = fullfile(outputdir,'second');
    checkdir(firstlvl_dir,secondlvl_dir)

    compare_metrics = {char(sprintf("%s_teststimpairs_gtlocEuclidean",subsets{j})),...
                       char(sprintf("%s_teststimpairs_feature2d",subsets{j}))};
    compare_metrics_indices = cellfun(@(x) find(strcmp(metric_names,x)), compare_metrics);
    
    
    for jsub = 1:numel(participants.validids)
        rdm_corr_coef_dir = fullfile(rsa_dir,'first',participants.validids{jsub});
        input_imgs = arrayfun(@(idx) char(fullfile(rdm_corr_coef_dir, sprintf("stransformed_rho_%04d.nii",idx-1))), compare_metrics_indices, 'uni', 0 );
        spm_imcalc(input_imgs,char(fullfile(firstlvl_dir,sprintf("%s_diff.nii",participants.validids{jsub}))),'i1-i2',struct("dtype",16))
    end
    
    all_fnames = cellfun(@(p) char(sprintf("%s_diff.nii",p)), participants.validids, 'uni', 0);
    g_fnames = cellfun(@(p) char(sprintf("%s_diff.nii",p)), participants.generalizerids, 'uni', 0);
    ng_fnames = cellfun(@(p) char(sprintf("%s_diff.nii",p)), participants.nongeneralizerids, 'uni', 0);
    
    scans = cellstr(fullfile(firstlvl_dir,all_fnames));
    scans_g = cellstr(fullfile(firstlvl_dir,g_fnames));
    scans_ng = cellstr(fullfile(firstlvl_dir,ng_fnames));
    
    glm_grouplevel(fullfile(secondlvl_dir,'allparticipants'),'t1',{scans},{comp_name});
    glm_estimate(fullfile(secondlvl_dir,'allparticipants'));
    glm_contrast(fullfile(secondlvl_dir,'allparticipants'),cellstr({"gtlocEuclidean>feature","feature>gtlocEuclidean"}),{[1],[-1]});
    
    % glm_grouplevel(fullfile(outputdir,'g_vs_ng'),'t2',{scans_g,scans_ng},{curr_metric});
    % glm_estimate(fullfile(outputdir,'g_vs_ng'));
    % glm_contrast(fullfile(outputdir,'g_vs_ng'), ...
    %             {strcat('G:',"gtlocEuclidean>feature"),strcat('NG:',"gtlocEuclidean>feature"),...
    %              strcat('G:',"feature>gtlocEuclidean"),strcat('NG:',"feature>gtlocEuclidean")}, ...
    %             {[1,0],[0,1], ...
    %              [-1,0],[0,-1]});
    %glm_results(fullfile(outputdir,'generalizer_vs_nongeneralizer'),1,struct('type','none','val',0.001,'extent',0),{'csv'});
    
    glm_grouplevel(fullfile(secondlvl_dir,'generalizer_only'),'t1',{scans_g},{comp_name});
    glm_estimate(fullfile(secondlvl_dir,'generalizer_only'));
    glm_contrast(fullfile(secondlvl_dir,'generalizer_only'),cellstr({"gtlocEuclidean>feature","feature>gtlocEuclidean"}),{[1],[-1]});
    
    
    glm_grouplevel(fullfile(secondlvl_dir,'nongeneralizer_only'),'t1',{scans_ng},{comp_name});
    glm_estimate(fullfile(secondlvl_dir,'nongeneralizer_only'));
    glm_contrast(fullfile(secondlvl_dir,'nongeneralizer_only'),cellstr({"gtlocEuclidean>feature","feature>gtlocEuclidean"}),{[1],[-1]});


end


%% Compare feature vs stimuligroup in Generalizers in all stimuli pairs
rsa_dir = fullfile(par_dirs{1},"correlation","all_nanidentity");
metric_names = fieldnames(loadjson(fullfile(rsa_dir,'first',participants.validids{1},'searchlight.json')).estimator.modelRDMs);

subsets = {"between", "within"};
for j = 1:numel(subsets)
    comp_name = sprintf("%s_allstimpairs_featvsSG",subsets{j});
    outputdir = fullfile(rsa_dir,"compare_rdm_corrcoefs",comp_name);
    firstlvl_dir = fullfile(outputdir,'first');
    secondlvl_dir = fullfile(outputdir,'second');
    checkdir(firstlvl_dir,secondlvl_dir)

    compare_metrics = {char(sprintf("%s_feature2d",subsets{j})),...
                       char(sprintf("%s_stimuligroup",subsets{j}))};
    compare_metrics_indices = cellfun(@(x) find(strcmp(metric_names,x)), compare_metrics);
    
    
    for jsub = 1:numel(participants.validids)
        rdm_corr_coef_dir = fullfile(rsa_dir,'first',participants.validids{jsub});
        input_imgs = arrayfun(@(idx) char(fullfile(rdm_corr_coef_dir, sprintf("stransformed_rho_%04d.nii",idx-1))), compare_metrics_indices, 'uni', 0 );
        spm_imcalc(input_imgs,char(fullfile(firstlvl_dir,sprintf("%s_diff.nii",participants.validids{jsub}))),'i1-i2',struct("dtype",16))
    end
    
    all_fnames = cellfun(@(p) char(sprintf("%s_diff.nii",p)), participants.validids, 'uni', 0);
    g_fnames = cellfun(@(p) char(sprintf("%s_diff.nii",p)), participants.generalizerids, 'uni', 0);
    ng_fnames = cellfun(@(p) char(sprintf("%s_diff.nii",p)), participants.nongeneralizerids, 'uni', 0);
    
    scans = cellstr(fullfile(firstlvl_dir,all_fnames));
    scans_g = cellstr(fullfile(firstlvl_dir,g_fnames));
    scans_ng = cellstr(fullfile(firstlvl_dir,ng_fnames));
    
    glm_grouplevel(fullfile(secondlvl_dir,'allparticipants'),'t1',{scans},{comp_name});
    glm_estimate(fullfile(secondlvl_dir,'allparticipants'));
    glm_contrast(fullfile(secondlvl_dir,'allparticipants'),cellstr({"feature>stimuligroup","stimuligroup>feature"}),{[1],[-1]});
    
        
    glm_grouplevel(fullfile(secondlvl_dir,'generalizer_only'),'t1',{scans_g},{comp_name});
    glm_estimate(fullfile(secondlvl_dir,'generalizer_only'));
    glm_contrast(fullfile(secondlvl_dir,'generalizer_only'),cellstr({"feature>stimuligroup","stimuligroup>feature"}),{[1],[-1]});
    
    
    glm_grouplevel(fullfile(secondlvl_dir,'nongeneralizer_only'),'t1',{scans_ng},{comp_name});
    glm_estimate(fullfile(secondlvl_dir,'nongeneralizer_only'));
    glm_contrast(fullfile(secondlvl_dir,'nongeneralizer_only'),cellstr({"feature>stimuligroup","stimuligroup>feature"}),{[1],[-1]});


end