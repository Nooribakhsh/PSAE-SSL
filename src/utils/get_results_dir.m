function [resultsDir, tag] = get_results_dir(projectRoot)
% GET_RESULTS_DIR  Results folder that matches the active settings.
%   The first settings used write their signature into results/settings_used.txt.
%   As long as the signature stays the same, results go to <root>/results.
%   If a result-relevant setting changes, results go to <root>/results_cfg<hash>
%   so runs made with different settings never mix.
%   (datasets, seeds, download options and sensitivity grids do not count:
%   per-seed caches are reused when the number of seeds grows.)
usr = get_user_settings();
skip = {'datasets','autoDownload','forceDownload','zenodoRecord','numSeeds', ...
        'mainRuns','statSeeds','statDatasets','branchValues','labelRatios', ...
        'labelSensDatasets'};
rel = rmfield(usr, skip);
str = jsonencode(rel);
h = mod(sum(double(str) .* (1:numel(str))), 16777216);
sig = sprintf('cfg%06x', h);

baseDir = fullfile(projectRoot, 'results');
sigFile = fullfile(baseDir, 'settings_used.txt');
tag = '';
resultsDir = baseDir;
if exist(sigFile, 'file')
    fid = fopen(sigFile, 'r');
    first = '';
    if fid > 0, first = strtrim(fgetl(fid)); fclose(fid); end
    if ischar(first) && ~strcmp(first, ['signature: ' sig])
        tag = sig;
        resultsDir = fullfile(projectRoot, ['results_' sig]);
    end
end
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end
outFile = fullfile(resultsDir, 'settings_used.txt');
if ~exist(outFile, 'file')
    fid = fopen(outFile, 'w');
    if fid > 0
        fprintf(fid, 'signature: %s\n', sig);
        fprintf(fid, '%s\n', str);
        fclose(fid);
    end
end
end
