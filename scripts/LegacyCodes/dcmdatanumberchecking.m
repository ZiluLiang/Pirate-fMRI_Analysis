raw_data_dir   = 'F:\MRI_Granada\CHRIS\ZILU';
sublist = {dir(raw_data_dir).name};
sublist = sublist(cellfun(@(x) ~contains(x,'.'),sublist));

counts = table();
for isub = 1:numel(sublist)
    subfolders = {dir(fullfile(raw_data_dir,sublist{isub},'Mruz_Chris')).name};
    subfolders = subfolders(cellfun(@(x) ~contains(x,'.'),subfolders));
    c = [sublist(isub),cellfun(@(sdir) numel({dir(fullfile(raw_data_dir,sublist{isub},'Mruz_Chris',sdir,'*.dcm')).name}),subfolders,'uni',0)];
    if isub>1
        t = cell2table(c,'VariableNames',["subid",subfolders]);
        counts = outerjoin(counts,t, 'MergeKeys',true);
    else
        counts = cell2table(c,'VariableNames',["subid",subfolders]);
    end
end

writetable(counts,'checkdatatable.xls','Sheet',"zilu")