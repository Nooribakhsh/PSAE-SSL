function ensure_datasets(projectRoot, datasets, usr)
% ENSURE_DATASETS  Make sure data/<name>/edges.txt and labels.txt exist.
%   Missing datasets are downloaded from the Zenodo record given in
%   get_user_settings.m (usr.zenodoRecord, default 10.5281/zenodo.18775800),
%   unpacked, and copied to data/<name>/.
%
%   ensure_datasets(projectRoot, datasets)          % datasets = cellstr
%   ensure_datasets(projectRoot, datasets, usr)     % explicit settings
%
%   The Zenodo record 18775800 contains ONE file, Datasets.rar. MATLAB cannot open
%   RAR archives itself, so an external extractor is used (7-Zip, WinRAR/UnRAR,
%   unrar, unar or bsdtar/tar - whichever is found). Install 7-Zip
%   (https://www.7-zip.org) if none is available.
%
%   Expected content after unpacking: for every dataset a folder (anywhere inside
%   the archive) named like the dataset (karate, dolphin(s), football, polbook(s),
%   polblog(s), email, cora, citeseer, pubmed, dblp) that contains edges.txt and
%   labels.txt in the layout of data/README.md (1-indexed node ids, line i of
%   labels.txt = community of node i). Any other layout is NOT converted
%   automatically (a wrong conversion would silently corrupt the results);
%   the function then stops and lists what it found.
if nargin < 3 || isempty(usr), usr = get_user_settings(); end
if ischar(datasets), datasets = {datasets}; end
dataRoot = fullfile(projectRoot, 'data');
if ~exist(dataRoot, 'dir'), mkdir(dataRoot); end

need = {};
for i = 1:numel(datasets)
    key = norm_name(datasets{i});
    if usr.forceDownload || ~dataset_ok(dataRoot, key)
        need{end+1} = key; %#ok<AGROW>
    end
end
need = unique(need, 'stable');
if isempty(need)
    fprintf('[data] all %d requested datasets found in %s\n', numel(datasets), dataRoot);
    return
end
fprintf('[data] missing/forced: %s\n', strjoin(need, ', '));
if ~usr.autoDownload
    error('ensure_datasets:missing', ['Datasets missing (%s) and usr.autoDownload = false.\n' ...
        'Download them from https://doi.org/10.5281/zenodo.%s and place them in %s'], ...
        strjoin(need, ', '), usr.zenodoRecord, dataRoot);
end

stage = fullfile(tempdir, ['psae_zenodo_' usr.zenodoRecord]);
if exist(stage, 'dir'), rmdir(stage, 's'); end
mkdir(stage);

%% ---- 1. list the files of the Zenodo record ----
apiUrl = ['https://zenodo.org/api/records/' usr.zenodoRecord];
fprintf('[data] querying %s\n', apiUrl);
optsJ = weboptions('Timeout', 120, 'ContentType', 'json');
try
    rec = webread(apiUrl, optsJ);
catch ME
    warning('ensure_datasets:api', 'Zenodo API not reachable (%s) - trying the direct file link.', ME.message);
    rec = struct('files', {{struct('key', 'Datasets.rar')}});
end
if isfield(rec, 'files') && ~isempty(rec.files)
    files = rec.files;
    if iscell(files) && numel(files) == 1 && iscell(files{1}), files = files{1}; end
else
    % newer Zenodo API: file list under /files -> .entries
    try
        Fl = webread([apiUrl '/files'], optsJ);
        if isfield(Fl, 'entries'), files = Fl.entries; else, files = []; end
    catch
        files = [];
    end
end
if isstruct(files), files = num2cell(files); end
if isempty(files)
    error('ensure_datasets:nofiles', 'The Zenodo record %s lists no files.', usr.zenodoRecord);
end

%% ---- 2. download + unpack every file ----
optsD = weboptions('Timeout', 3600);
for i = 1:numel(files)
    f = files{i};
    fname = pick_field(f, {'key','filename','name'});
    url = '';
    if isfield(f, 'links') && isstruct(f.links)
        url = pick_field(f.links, {'self','content','download'});
    end
    if isempty(url)
        url = sprintf('https://zenodo.org/records/%s/files/%s?download=1', usr.zenodoRecord, fname);
    end
    target = fullfile(stage, regexprep(fname, '[\\/]', '_'));
    fprintf('[data] downloading %s ...\n', fname);
    websave(target, url, optsD);
    unpack(target, stage);
end

%% ---- 2b. unpack any nested zip/tar archives (Zenodo RAR contains *.zip) ----
% The Zenodo archive is a RAR of per-dataset ZIPs. After extracting the RAR,
% each ZIP still needs to be opened so that edges.txt / labels.txt appear.
nested = [dir(fullfile(stage, '**', '*.zip'));
          dir(fullfile(stage, '**', '*.tar'));
          dir(fullfile(stage, '**', '*.tar.gz'));
          dir(fullfile(stage, '**', '*.tgz'))];
for i = 1:numel(nested)
    if nested(i).isdir, continue; end
    zpath = fullfile(nested(i).folder, nested(i).name);
    % Extract into a subfolder named after the archive (minus extension)
    [~, zbase, ~] = fileparts(nested(i).name);
    zbase = strtrim(zbase);   % handle names like "Polblogs .zip"
    zdest = fullfile(nested(i).folder, zbase);
    if ~exist(zdest, 'dir'), mkdir(zdest); end
    fprintf('[data] unpacking nested %s -> %s\n', nested(i).name, zdest);
    try
        unpack(zpath, zdest);
    catch ME
        warning('ensure_datasets:nested', 'Could not unpack nested %s: %s', zpath, ME.message);
    end
end

%% ---- 3. find the dataset folders and copy them to data/ ----
E = dir(fullfile(stage, '**', 'edges.txt'));
found = struct('key', {}, 'folder', {});
for i = 1:numel(E)
    if exist(fullfile(E(i).folder, 'labels.txt'), 'file')
        [~, leaf] = fileparts(E(i).folder);
        found(end+1) = struct('key', norm_name(leaf), 'folder', E(i).folder); %#ok<AGROW>
    end
end
% Also accept edges/labels sitting directly under a zip-named folder whose
% parent leaf already matched (e.g. stage/Karate/edges.txt).
if isempty(found)
    % Fallback: look one level deeper if labels live next to edges under any name
    for i = 1:numel(E)
        if exist(fullfile(E(i).folder, 'labels.txt'), 'file')
            [~, leaf] = fileparts(E(i).folder);
            found(end+1) = struct('key', norm_name(leaf), 'folder', E(i).folder); %#ok<AGROW>
        end
    end
end
for i = 1:numel(need)
    idx = find(strcmp({found.key}, need{i}), 1);
    if isempty(idx)
        error('ensure_datasets:notfound', ['Dataset "%s" was not found in Zenodo record %s.\n' ...
            'Folders with edges.txt + labels.txt found there: %s\n' ...
            'Unpacked files are in %s - copy the right folder to %s manually.'], ...
            need{i}, usr.zenodoRecord, strjoin({found.key}, ', '), stage, fullfile(dataRoot, need{i}));
    end
    dest = fullfile(dataRoot, need{i});
    if exist(dest, 'dir'), rmdir(dest, 's'); end
    copyfile(found(idx).folder, dest);
    fprintf('[data] installed %s\n', dest);
end
try rmdir(stage, 's'); catch, end %#ok<NOCOM>
fprintf('[data] done.\n');
end

function ok = dataset_ok(dataRoot, key)
ok = exist(fullfile(dataRoot, key, 'edges.txt'), 'file') && ...
     exist(fullfile(dataRoot, key, 'labels.txt'), 'file');
if ~ok && strcmp(key, 'dolphins')      % legacy folder name
    ok = exist(fullfile(dataRoot, 'dolphin', 'edges.txt'), 'file') && ...
         exist(fullfile(dataRoot, 'dolphin', 'labels.txt'), 'file');
end
end

function key = norm_name(name)
key = lower(regexprep(strtrim(name), '[^A-Za-z0-9]', ''));
alias = {'dolphin','dolphins'; 'polbook','polbooks'; 'polblog','polblogs'; ...
         'emaileu','email'; 'emaileucore','email'; 'cite','citeseer'};
k = find(strcmp(key, alias(:,1)), 1);
if ~isempty(k), key = alias{k,2}; end
end

function v = pick_field(s, names)
v = '';
for i = 1:numel(names)
    if isfield(s, names{i}) && ~isempty(s.(names{i}))
        v = s.(names{i}); return
    end
end
end

function unpack(file, outDir)
[~, ~, ext] = fileparts(file);
lf = lower(file);
try
    if strcmp(ext, '.zip')
        unzip(file, outDir);
    elseif endsWith(lf, '.tar.gz') || endsWith(lf, '.tgz')
        untar(file, outDir);
    elseif strcmp(ext, '.tar')
        untar(file, outDir);
    elseif strcmp(ext, '.gz')
        gunzip(file, outDir);
    elseif strcmp(ext, '.rar')
        unrar_file(file, outDir);
    end
catch ME
    error('ensure_datasets:unpack', ['Could not unpack %s:\n%s\n' ...
        'The downloaded file is kept in %s - unpack it manually (copy the dataset folders ' ...
        'with edges.txt/labels.txt into your data/ folder) and run again.'], file, ME.message, outDir);
end
end

function unrar_file(file, outDir)
% Extract a RAR archive with whatever external tool is available.
if ispc
    cand = {'C:\Program Files\7-Zip\7z.exe', 'C:\Program Files (x86)\7-Zip\7z.exe', ...
            'C:\Program Files\WinRAR\UnRAR.exe', 'C:\Program Files (x86)\WinRAR\UnRAR.exe', ...
            '7z', '7za', 'unrar', 'bsdtar', 'tar'};
else
    cand = {'7z', '7za', '7zr', 'unrar', 'unar', 'bsdtar', 'tar'};
end
tried = {};
for i = 1:numel(cand)
    exe = cand{i};
    if any(exe == filesep) || any(exe == ':')
        if ~exist(exe, 'file'), continue; end
    else
        if ispc, [st, ~] = system(['where ' exe ' >nul 2>nul']);
        else,    [st, ~] = system(['which ' exe ' >/dev/null 2>&1']); end
        if st ~= 0, continue; end
    end
    [~, base] = fileparts(exe); base = lower(base);
    if any(strcmp(base, {'7z','7za','7zr'}))
        cmd = sprintf('"%s" x -y -o"%s" "%s"', exe, outDir, file);
    elseif strcmp(base, 'unrar')
        cmd = sprintf('"%s" x -o+ -y "%s" "%s%s"', exe, file, outDir, filesep);
    elseif strcmp(base, 'unar')
        cmd = sprintf('"%s" -o "%s" "%s"', exe, outDir, file);
    else   % bsdtar / tar (libarchive reads RAR)
        cmd = sprintf('"%s" -xf "%s" -C "%s"', exe, file, outDir);
    end
    fprintf('[data] extracting with %s ...\n', exe);
    [st, msg] = system(cmd);
    if st == 0, return; end
    tried{end+1} = sprintf('%s: %s', exe, strtrim(msg)); %#ok<AGROW>
end
if isempty(tried)
    error('ensure_datasets:norar', ['No RAR extractor found. Install 7-Zip (https://www.7-zip.org) or ' ...
        'WinRAR and run again.']);
else
    error('ensure_datasets:rarfail', 'RAR extraction failed:\n%s', strjoin(tried, newline));
end
end
