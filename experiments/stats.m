%% Summary statistics of PSAE-SSL: mean +/- std per dataset and metric.
% Reads results/collected.mat (run collect_results first) and prints one table
% (also saved to results/tables/psae_summary.csv). Baseline comparisons
% (Friedman / Wilcoxon) are in statistical_tests.m.
clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
cf = fullfile(resultsDir, 'collected.mat');
if ~exist(cf, 'file')
    error('%s not found. Run all_datasets (or nmi_q_log) and collect_results first.', cf);
end
C = load(cf);
tablesDir = fullfile(resultsDir, 'tables');
if ~exist(tablesDir, 'dir'); mkdir(tablesDir); end
fid = fopen(fullfile(tablesDir, 'psae_summary.csv'), 'w');
fprintf(fid, 'dataset,%s\n', strjoin(strcat(C.metricNames, '_mean,', C.metricNames, '_std'), ','));
fprintf('%-10s', 'Dataset'); fprintf(' %16s', C.metricNames{:}); fprintf('\n');
for di = 1:numel(C.datasets)
    fprintf('%-10s', C.datasets{di}); fprintf(fid, '%s', C.datasets{di});
    for m = 1:5
        mu = C.meanT(1, di, m); sd = C.stdT(1, di, m);
        fprintf(' %7.4f+-%-7.4f', mu, sd); fprintf(fid, ',%.4f,%.4f', mu, sd);
    end
    fprintf('\n'); fprintf(fid, '\n');
end
fclose(fid);
fprintf('\nSaved results/tables/psae_summary.csv\n');
