function f = check_head_motion(subimg_dir,varargin)
% show the animated 4D series (preprocessed fmri niis) to check the effect of
% motion correction using MRIcroGl.
% plot the head motion parameters estimated during motion correction (realign)
% usage: checkHeadMotion(subimg_dir,flag_animate,flag_plotmotion,flag_showmotion)
    if numel(varargin)>3
        error('Too many input arguments')
    end
    
    % get the filepattern from default setting 
    filepattern = get_pirate_defaults(false,'filepattern');
    
    %set flags
    flags = {1,1,1};
    flags(1:numel(varargin)) = varargin(:);
    [flag_animate,flag_plotmotion,flag_showmotion] = flags{:};
    
    %find files
    img_files = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.normalise,'.*.nii']));
    rp_files  = cellstr(spm_select('FPList',subimg_dir,[filepattern.preprocess.motionparam,'.*.txt']));
    
    if flag_animate
        view4DAnimation(img_files,rp_files)
    end
    
    if flag_plotmotion
        f = plot_rp(rp_files,flag_showmotion);
    end    
end

function f = plot_rp(rp_files,flag_showmotion)
    filepattern = get_pirate_defaults(false,'filepattern');
    tablenames = {'Displacement from first frame',...
                'Framewise displacement',};
    yvars = {'x','y','z','pitch','yaw','roll'};
    n_sessions = numel(rp_files);
    
    f = figure;
    if ~flag_showmotion
        set(gcf,'Visible', 'off');
    end
    tiledlayout(n_sessions,2)
    
    for j = 1:n_sessions
        [~,task_run,~] = fileparts(rp_files{j});
        task_run = regexprep(task_run,filepattern.preprocess.motionparam,'');        

        displacement = readtable(rp_files{j});        
        displacement.Properties.VariableNames = yvars;
        framewisedisplacement = array2table(table2array(displacement(2:end,:)) - table2array(displacement(1:end-1,:)),'VariableNames',yvars);
        tables = {displacement,framewisedisplacement};
        
        for t = 1:numel(tables)
            nexttile
            for k = 1:numel(yvars)
                plot(1:size(tables{t},1),tables{t}.(yvars{k}))
                hold on
            end
            title(strjoin({tablenames{t},task_run},'\n'),'Interpreter','none')
        end
    end
    hL = legend(yvars{:});
    hL.Layout.Tile = 'North';
    hL.Orientation = 'horizontal';
    f.Position = get(0, 'ScreenSize');
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
