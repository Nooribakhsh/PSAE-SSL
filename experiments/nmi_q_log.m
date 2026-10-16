%% Log NMI + Modularity (Q) + P/R/F1 for all available datasets
% Protocol: 10 seeds, paper hyperparameters. Prints a clear table at the end.

clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

usr       = get_user_settings();   % settings: src/utils/get_user_settings.m
numSeeds  = usr.numSeeds;
labelFrac = usr.labelFrac;
datasets  = usr.datasets;
ensure_datasets(projectRoot, datasets, usr);
metricNames = {'NMI','Q','Precision','Recall','F1'};

fprintf('=== NMI / Q log — %d seeds, datasets: %s ===\n', numSeeds, strjoin(datasets, ', '));

allMeans = nan(numel(datasets), 5);
allStds  = nan(numel(datasets), 5);

for di = 1:numel(datasets)
    name = datasets{di};
    folder = fullfile(projectRoot, 'data', name);
    if ~exist(folder, 'dir')
        warning('[%s] missing data folder, skip', name); continue
    end
    cfg = get_dataset_config(name);
    [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
    n = numel(adjList);
    fprintf('\n--- %s (n=%d, K=%d) ---\n', name, n, K);

    scores = nan(numSeeds, 5);
    for s = 1:numSeeds
        try
            r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                struct('labelFrac', labelFrac, 'stratified', false, 'B', cfg.B));
            scores(s,:) = [r.NMI, r.Q, r.Precision, r.Recall, r.F1];
            fprintf('  seed=%2d  NMI=%.4f  Q=%.4f  P=%.4f  R=%.4f  F1=%.4f  (bestK=%d)\n', ...
                s, scores(s,:), r.bestK);
        catch ME
            warning('  seed=%d FAILED: %s', s, ME.message);
        end
    end
    allMeans(di,:) = mean(scores, 1, 'omitnan');
    allStds(di,:)  = std(scores, 0, 1, 'omitnan');
    save(fullfile(resultsDir, ['nmi_q_' name '.mat']), 'name', 'scores', 'allMeans', 'allStds');
end

save(fullfile(resultsDir, 'nmi_q_summary.mat'), 'datasets', 'allMeans', 'allStds', 'metricNames', 'numSeeds');
fprintf('\nSaved results/nmi_q_summary.mat\n');
