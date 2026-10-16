%% Generate simple text/LaTeX summary tables for PSAE-SSL only.
% Needs results/collected.mat (from collect_results).
% Output: results/tables/tbl_psae_ssl_main.tex and .txt

clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
outDir = fullfile(resultsDir, 'tables');
if ~exist(outDir, 'dir'); mkdir(outDir); end
cf = fullfile(resultsDir, 'collected.mat');
if ~exist(cf, 'file')
    error('Run collect_results first.');
end
C = load(cf);
ds = C.datasets;
met = C.metricNames;

% text table
fid = fopen(fullfile(outDir, 'tbl_psae_ssl_main.txt'), 'w');
fprintf(fid, '%-12s', 'Dataset');
for m = 1:5, fprintf(fid, ' %14s', met{m}); end
fprintf(fid, '\n');
for di = 1:numel(ds)
    fprintf(fid, '%-12s', ds{di});
    for m = 1:5
        mu = C.meanT(1, di, m); sd = C.stdT(1, di, m);
        if isnan(mu), fprintf(fid, ' %14s', 'n/a');
        else, fprintf(fid, ' %6.4f±%-6.4f', mu, sd); end
    end
    fprintf(fid, '\n');
end
fclose(fid);

% LaTeX tabular
L = {};
L{end+1} = '\begin{tabular}{lrrrrr}';
L{end+1} = '\toprule';
L{end+1} = ['Dataset & ' strjoin(met, ' & ') ' \\'];
L{end+1} = '\midrule';
for di = 1:numel(ds)
    cells = cell(1, 5);
    for m = 1:5
        mu = C.meanT(1, di, m); sd = C.stdT(1, di, m);
        if isnan(mu), cells{m} = 'n/a';
        else, cells{m} = sprintf('%.4f $\\pm$ %.4f', mu, sd); end
    end
    L{end+1} = [ds{di} ' & ' strjoin(cells, ' & ') ' \\'];
end
L{end+1} = '\bottomrule';
L{end+1} = '\end{tabular}';
fid = fopen(fullfile(outDir, 'tbl_psae_ssl_main.tex'), 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('Wrote results/tables/tbl_psae_ssl_main.txt and .tex\n');
