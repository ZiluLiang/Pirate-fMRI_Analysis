clear;clc

[directory,participants,filepattern]  = get_pirate_defaults(false,'directory','participants','filepattern');
masks = cellstr(spm_select('FPList','E:\pirate_fmri\Analysis\data\Exp1_fmri\fmri\masks\anat','.*.nii'));
err_tracker   = struct(); %#ok<*UNRCH>
addpath(fullfile(directory.projectdir,'scripts','Exp1_fmri','univariate'))
        
glm_name = 'traintest_navigation';        
glm_dir  = fullfile(directory.fmri_data,glm_name);

reg_prim_names = glm_configure(glm_name).conditions;

pgroups = struct('C1C2', {participants.validids},...
                 'C1',{participants.cohort1ids}, ...
                 'C2',{participants.cohort2ids});
gnames = fieldnames(pgroups);
curr_g = 'C1C2';
subid_list = participants.validids(cellfun(@(x) ismember(x,pgroups.(curr_g)), participants.validids));
gzer_list  = participants.generalizerids(cellfun(@(x) ismember(x, subid_list), participants.generalizerids));
ngzer_list = participants.nongeneralizerids(cellfun(@(x) ismember(x, subid_list), participants.nongeneralizerids));

whole_des_orths = cell(numel(subid_list),1);
reg_des_orths = cell(numel(subid_list),4);
for isub = 1:numel(subid_list)
    firstlvl_spmdir = fullfile(glm_dir,"first",subid_list{isub},"SPM.mat");
    SPM = load(firstlvl_spmdir).SPM;
    whole_des_orths{isub} = abs(1-squareform(pdist(SPM.xX.xKXs.X','cosine'))); 
    
    % only look at condition columns
    reg_filters =cellfun(@(y) cellfun(@(x) contains(x,y),SPM.xX.name),reg_prim_names,'uni',0);
    for r = 1:4
        reg_sess_name = cellfun(@(x) sprintf('Sn(%d) %s',r,x), reg_prim_names,'uni',0);
        reg_filters = cellfun(@(y) cellfun(@(x) contains(x,y),SPM.xX.name),reg_sess_name,'uni',0);
        reg_filters = sum(cat(1,reg_filters{:}))>0;
        regcols = SPM.xX.xKXs.X(:,reg_filters);
        reg_des_orths{isub,r} = abs(1-squareform(pdist(regcols','cosine'))); 
    end
end
stack_reg_des_orths = arrayfun(@(x) cat(3,reg_des_orths{x,:}),1:numel(subid_list),'uni',0);

figure 
tiles =  tiledlayout(6,10);
tiles.TileSpacing = 'loose';
for isub=1:numel(subid_list)
    %subplot(4,14,isub)
    currax = nexttile(tiles);
    xticks(''); yticks(''); xticklabels(''); yticklabels('');
    daspect(currax,[1,1,1])
    hdata = tril(max(stack_reg_des_orths{isub},[],3),-1);
    hdata(hdata==0) = nan;
    heatmap(hdata,'Colormap',summer);
    if ismember(subid_list{isub},gzer_list)
        sg = "G";
    else
        sg = "nG";
    end
    title(sprintf('%s-%s',sg,subid_list{isub}))
    %axis equal
end
