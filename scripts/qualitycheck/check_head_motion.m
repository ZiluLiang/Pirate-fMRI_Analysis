function [T,f] = check_head_motion(subimg_dir,rp_range,rp_thres,flag_animate,flag_plotmotion,flag_showmotion)
% show the animated 4D series (preprocessed fmri niis) to check the effect of
% motion correction using MRIcroGl.
% plot the head motion parameters estimated during motion correction (realign)
% usage: check_head_motion(subimg_dir,rp_range,rp_thres,flag_animate,flag_plotmotion,flag_showmotion)
% INPUTS:
%     - subimg_dir: directory that contains the rp file and preprocessed
%     image of participants
%     - rp_range: 1x2 numeric array ([min,max]), specifying range of the yaxis in rp plots
%     - rp_thres: number, specifying threshold for excessive head movement
%     - flag_animate: show animated 4D series of preprocessed data in MRIcroGL or not
%     - flag_plotmotion: produce line plot of the headmotion parameters or not
%     - flag_showmotion: show line plot of the headmotion parameters or not
% OUTPUTS:
%     - T: table of head motion metrics
%     - f: the figure handle of the head motion plots if flag_plotmotion is
%     true, else return nan
% -----------------------------------------------------------------------    
% Author: Zilu Liang

    % get the filepattern from default setting 
    filepattern = get_pirate_defaults(false,'filepattern');
    
    %set flags
    if nargin<5, flag_showmotion = 1; end
    if nargin<4, flag_plotmotion = 1; end
    if nargin<3, flag_animate = 1; end
    if nargin<2, rp_range = []; end
    
    %find files
    img_files = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.normalise,'.*.nii']));
    rp_files  = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.motionparam,'.*.txt']));
    
    if flag_animate
        view4DAnimation(img_files,rp_files)
    end
    
    n_sessions = numel(rp_files);
    tasknames = cell(n_sessions,1);
    % tables for plotting
    rp_tables = cell(n_sessions,2);
    tablenames = {'Displacement from first frame',...
                  'Framewise displacement',};
    yvars = {'x','y','z','pitch','yaw','roll'};
    % summary tables of head motion
    metrics = {'max absolute displacement from first frame(mm)',...
               'number of outliers - displacement from first frames',...
               'percentage of outliers - displacement from first frames (%)',...
               'mean absolute framewise displacement(mm)',...
               'number of outliers - framewise displacement',...
               'percentage of outliers - framewise displacement (%)'};
    T = nan(n_sessions,numel(metrics));    
    for j = 1:n_sessions
        [~,tasknames{j},~] = fileparts(rp_files{j});
        tasknames{j} = regexprep(tasknames{j},filepattern.preprocess.motionparam,'');
        
        % displacement(D)
        D = readtable(rp_files{j});        
        D.Properties.VariableNames = yvars;
        % framewise displacement(FW)
        FW = array2table(table2array(D(2:end,:)) - table2array(D(1:end-1,:)),'VariableNames',yvars);
        % convert rotational displacements from radians to millimeters by calculating displacement on the surface of a sphere of radius 50 mm
        D(:,4:6)  = array2table(table2array(D(:,4:6))*50,'VariableNames',yvars(4:6));        
        FW(:,4:6) = array2table(table2array(FW(:,4:6))*50,'VariableNames',yvars(4:6));       
        rp_tables{j,1} = D;
        rp_tables{j,2} = FW;
        % summaries of metrics
        sum_FW = rowfun(@(varargin) sum(cellfun(@(x) abs(x),varargin),'all'),FW,'OutputFormat','uniform');
        sum_D  = rowfun(@(varargin) sum(cellfun(@(x) abs(x),varargin),'all'),D,'OutputFormat','uniform');
        T(j,:) = [max(sum_D),...
                  sum(sum_D>rp_thres),...
                  sum(sum_D>rp_thres)/numel(sum_D),...
                  mean(sum_FW),...
                  sum(sum_FW>rp_thres),...
                  sum(sum_FW>rp_thres)/numel(sum_FW)];
    end
    T = array2table(T,'VariableNames',metrics);
    
    if flag_plotmotion
        f = plot_rp(rp_tables,tasknames,tablenames,rp_range,rp_thres,flag_showmotion);
    else
        f = nan;
    end    
end


function f = plot_rp(rp_tables,tasknames,tablenames,rp_range,rp_thres,flag_showmotion)
    yvars = {'x','y','z','pitch','yaw','roll'};
    n_sessions = size(rp_tables,1);
    
    f = figure;
    if ~flag_showmotion
        set(gcf,'Visible', 'off');
    end
    tiledlayout(size(rp_tables,1),size(rp_tables,2))
    
    for j = 1:n_sessions
        for t = 1:numel(tablenames)
            nexttile
            for k = 1:numel(yvars)
                plot(1:size(rp_tables{j,t},1),rp_tables{j,t}.(yvars{k}))
                yline(rp_thres,'r--')
                yline(-rp_thres,'r--')
                ylim(rp_range)
                hold on
            end
            title(strjoin({tablenames{t},tasknames{j}},'\n'),'Interpreter','none')
        end
    end
    hL = legend(yvars{:});
    hL.Layout.Tile = 'North';
    hL.Orientation = 'horizontal';
    f.Position = get(0, 'ScreenSize');
    sgtitle(sprintf('rotational displacements converted from radians to mm \n by calculating displacement on the surface of a 50mm-radius sphere'),...
        'FontSize',8)
end

function view4DAnimation(img_files,rp_files)
    [parent_dir,~,~] = fileparts(img_files{1});
    scriptfile = fullfile(parent_dir,'view4Dseries.py');
    scriptfileID = fopen(scriptfile,'w');
    path_MRIcroGL = 'C:\MRIcroGL_windows\MRIcroGL\MRIcroGL.exe';
    
    n_volumes = cellfun(@(fn) size(spm_vol(fn),1),img_files);

    fmt = strjoin(["import gl",...
                   "img_files = [%s]",...
                   "rp_files  = [%s]",...
                   "n_volumes = [%s]",...
                   "for img_fn,rp_fn,nvol in zip(img_files,rp_files,n_volumes):",... 
                   "  gl.loadimage(img_fn)",...
                   "  gl.loadgraph(rp_fn)",...
                   "  for j in range(nvol):",...
                   "    gl.volume(0,j)",...
                   "    gl.wait(10)",...
                   "  gl.modalmessage('Continue?')",...
                   "quit()"],"\n");
    fprintf(scriptfileID,fmt,...
             strjoin(["r'",strjoin(img_files,"',r'"),"'"],""),...
             strjoin(["r'",strjoin(rp_files,"',r'"),"'"],""),...
             strjoin(arrayfun(@(x) num2str(x),n_volumes,'uni',0),","));
    fclose(scriptfileID);

    cmd=['"',path_MRIcroGL,'" "',scriptfile,'"'];
    system(cmd);
end
