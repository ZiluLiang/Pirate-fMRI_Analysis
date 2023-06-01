% %% examine FIR
% glm_name = 'sc_navigation';
% fir_tc = struct();
% fir_tc_c = struct();
% [fir_tc.(glm_name),fir_tc_c.(glm_name)] = get_fir(glm_name);
% 
% glm_name = 'sc_localizer';
% fir_tc = struct();
% fir_tc_c = struct();
% [fir_tc.(glm_name),fir_tc_c.(glm_name)] = get_fir(glm_name);
% 
% 
% %% plot fir filtered response
% glm_name = 'sc_navigation';
% line_colors = {'#0072BD','#D95319','#EDB120','#77AC30','#000000'};
% f = figure;
% tiledlayout(8,4)
% for j=1:numel(participants.validids)
%     nexttile
%     x = 1:size(fir_tc.(glm_name){j},1);
%     for irun = 1:4
%         y = fir_tc.(glm_name){j}(:,irun*2-1);
%         plot(x,y,'Color',line_colors{irun},'LineStyle','-','LineWidth',0.5)
%         hold on
%     end
%     %y = fir_tc_c.(glm_name){j}(:,1);
%     %plot(x,y,'Color',line_colors{5},'LineStyle','-','LineWidth',0.8)
%     xlabel('TR(1.73s)')
%     title(sprintf('sub-%d',j))    
% end
% line_names = [arrayfun(@(x) sprintf('run-%d',x),1:4,'uni',0),{'across runs'}];
% hL = legend(line_names);
% hL.Layout.Tile = 'North';
% hL.Orientation = 'horizontal';
% f.Position = get(0, 'ScreenSize');
% sgtitle('Percentage signgal change from stimuli onset by runs in AAL occipital regions')
% saveas(f,fullfile(directory.fmri_data,glm_name,'meanFIRperrun_stimuli_onset.png'));
% close(f)
% 
% 
% glm_name = 'sc_localizer';
% f = figure;
% tiledlayout(8,4)
% for j=1:numel(participants.validids)
%     nexttile
%     x = 1:size(fir_tc_c.(glm_name){j},1);
%     y1 = fir_tc_c.(glm_name){j}(:,1);
%     plot(x,y1,'k')
%     xlabel('TR(1.73s)')
%     title(sprintf('sub-%d',j))
%     hold on
% end
% hL = legend({'stimuli','motor'});
% hL.Layout.Tile = 'North';
% hL.Orientation = 'horizontal';
% f.Position = get(0, 'ScreenSize');
% sgtitle('Percentage signgal change from stimuli onset by runs in AAL occipital regions')
% saveas(f,fullfile(directory.fmri_data,glm_name,'meanFIR_stimuli_onset.png'));
% close(f)

% function [meanCon,meanResMS,maxStat] = extract_firstlvl_spmStat(glm_name)
%     [directory,participants]  = get_pirate_defaults(false,'directory','participants');
%     masks = {fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','AAL_Occipital.nii'),...
%              fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','HMAT_Motor.nii')};
%     glm_dir  = fullfile(directory.fmri_data,glm_name);
%     glm_config = get_glm_config(glm_name); 
%     maxStat = nan(numel(participants.validids),numel(glm_config.contrasts));
%     meanCon = nan(numel(participants.validids),numel(glm_config.contrasts));
%     for isub = 1:numel(participants.validids)
%         for j = 1:numel(glm_config.contrasts)
%             [~,con_img,stat_img] = find_contrast_idx(fullfile(glm_dir,'first',participants.validids{isub},'SPM.mat'),glm_config.contrasts(j).name);
%             voxelwise_Stat  = spm_summarise(fullfile(glm_dir,'first',participants.validids{isub},stat_img),masks{j});
%             voxelwise_Con   = spm_summarise(fullfile(glm_dir,'first',participants.validids{isub},con_img),masks{j});
%             voxelwise_ResMS = spm_summarise(fullfile(glm_dir,'first',participants.validids{isub},'ResMS.nii'),masks{j});            
%             maxStat(isub,j) = max(voxelwise_Stat);
%             meanCon(isub,j) = mean(voxelwise_Con,'all','omitnan');
%             meanResMS(isub,j) = mean(voxelwise_ResMS,'all','omitnan');            
%         end
%     end
%     maxStat = array2table(maxStat,'VariableNames',{glm_config.contrasts.name});
%     meanCon = array2table(meanCon,'VariableNames',{glm_config.contrasts.name});
%     meanResMS = array2table(meanResMS,'VariableNames',{glm_config.contrasts.name});
% end
% 
% function [fir_tc,fir_tc_c] = get_fir(glm_name)
%     [directory,participants]  = get_pirate_defaults(false,'directory','participants');
%     masks = {fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','aal_occipital_roi.mat'),...
%              fullfile('D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\data\fmri\masks','aal_occipital_roi.mat')};
%     glm_dir  = fullfile(directory.fmri_data,glm_name);
%     fir_tc   = cell(numel(participants.validids),1);
%     fir_tc_c = cell(numel(participants.validids),1);
%     for isub = 1:numel(participants.validids)
%         spmMAT = fullfile(glm_dir,'first',participants.validids{isub},'SPM.mat');
%         mask = masks{1};
%         % Make marsbar design object
%         D  = mardo(spmMAT);
%         % Make ROI 
%         R  = maroi(mask);
%         % Fetch data into marsbar data object
%         Y  = get_marsy(R, D, 'mean');
%         E = estimate(D, Y);
%         % Get definitions of all events in model
%         [e_specs, ~] = event_specs(E);
%         n_events = size(e_specs, 2);
%         % Bin size in seconds for FIR
%         bin_size = tr(E);
%         % Length of FIR in seconds
%         fir_length = 24;
%         % Number of FIR time bins to cover length of FIR
%         bin_no = fir_length / bin_size;
%         % Options - here 'single' FIR model, return estimated
%         opts = struct('single', 1,'percent',1);
%         % Return time courses for all events in fir_tc matrix
%         for e_s = 1:n_events
%           fir_tc{isub}(:, e_s) = event_fitted_fir(E, e_specs(:,e_s), bin_size, ...
%                                          bin_no, opts);
%         end
%         % Get compound event types structure
%         ets = event_types_named(E);
%         n_event_types = length(ets);
%         for e_t = 1:n_event_types
%            fir_tc_c{isub}(:, e_t) = event_fitted_fir(E, ets(e_t).e_spec, bin_size, ...
%               bin_no, opts);
%         end
%     end
% end