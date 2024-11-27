%should be laura.urbanosoto (sz6HzZaTXgvT) but ran the sequence of
%lautaronlcg(uT3IUrrac2yW) instead
datadir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\Analysis\fmri\data\fmri_behavior';
seq_dir = 'D:\OneDrive - Nexus365\Project\pirate_fmri\task\parameters';
exptid  = 'sz6HzZaTXgvT';
%% get randomization
correct_stimlist = readtable(fullfile(seq_dir,'param_ptb','stimlist_sz6HzZaTXgvT.txt'));

%% recode maintask
clear data meta
data_file = cellstr(spm_select('FPList',datadir,'sz6HzZaTXgvTmaintask_.*.mat'));
metadata_file = cellstr(spm_select('FPList',datadir,'sz6HzZaTXgvTmeta_maintask_.*.mat'));
keepvar = 'stim_img';
replacevars = {'stim_id','stim_attrx','stim_attrx','stim_x','stim_y','training'};

for irun = 1:4
    load(data_file{1},'data');
    load(metadata_file{1},'meta');
    %%%%%%for main task need to rely on stim_img and find the right stim_attrx
    %%%%%%and stim_x stim_y
    for j = 1:size(data,1)
        for k = 1:numel(replacevars)
            findrow = strcmp(correct_stimlist.(keepvar),data.(keepvar){j});
            data.(replacevars{k})(j) = correct_stimlist.(replacevars{k})(findrow);
        end
    end
    
    dist_lvl = [[1/3,2/3,1] * meta.displays.err_tol,Inf];
    bonus_level = [5,2,1,0]; 
    data.resp_acclvl = nan(size(data,1),1);
    data.bonus = zeros(size(data,1),1);
    
    data.resp_dist = arrayfun(@(j) norm([data.resp_x(j),data.resp_y(j)]-[data.stim_x(j),data.stim_y(j)]),1:size(data,1))';
    data.resp_acclvl(isnan(data.resp_dist)) = nan(sum(isnan(data.resp_dist)),1);
    data.resp_acclvl(~isnan(data.resp_dist)) = arrayfun(@(d) find(d<dist_lvl,1,'first'),data.resp_dist(~isnan(data.resp_dist)))';
    data.bonus(isnan(data.resp_dist)) = zeros(sum(isnan(data.resp_dist)),1);
    data.bonus(~isnan(data.resp_dist)) = bonus_level(data.resp_acclvl(~isnan(data.resp_dist)));

    [filepath,oldfn,ext] = fileparts(data_file{irun});
    parts = strsplit(oldfn,'_');

    newmatfn = sprintf('maintask_%s_%s_%s_%s%s',exptid,parts{3:5},ext);
    save(fullfile(filepath,newmatfn),'data')

    newtxtfn = sprintf('maintask_%s_%s_%s_%s%s',exptid,parts{3:5},'.txt');
    writetable(data,fullfile(filepath,newtxtfn),'Delimiter',',')

    newmetafn = sprintf('meta_maintask_%s_%s_%s_%s%s',exptid,parts{3:5},'.txt');
    save(fullfile(filepath,newmetafn),'meta')
end

%% recode localizer
clear data meta
data_file = cellstr(spm_select('FPList',datadir,'sz6HzZaTXgvTlocalizer_.*.mat'));
metadata_file = cellstr(spm_select('FPList',datadir,'sz6HzZaTXgvTmeta_localizer_.*.mat'));

load(data_file{1},'data')
load(metadata_file{1},'meta')

replacevars = 'stim_img';
keepvar = {'stim_id','stim_attrx','stim_attrx','stim_x','stim_y','training'};
for j = 1:size(data,1)
    criteria = cellfun(@(col) correct_stimlist.(col) == data.(col)(j),keepvar,'UniformOutput',0);
    findrow = find(all(cat(2,criteria{:}),2));
    data.(replacevars)(j) = correct_stimlist.(replacevars)(findrow);
end

[filepath,oldfn,ext] = fileparts(data_file{1});
parts = strsplit(oldfn,'_');

newmatfn = sprintf('localizer_%s_%s_%s_%s%s',exptid,parts{3:5},ext);
save(fullfile(filepath,newmatfn),'data')

newtxtfn = sprintf('localizer_%s_%s_%s_%s%s',exptid,parts{3:5},'.txt');
writetable(data,fullfile(filepath,newtxtfn),'Delimiter',',')

newmetafn = sprintf('meta_localizer_%s_%s_%s_%s%s',exptid,parts{3:5},'.txt');
save(fullfile(filepath,newmetafn),'meta')