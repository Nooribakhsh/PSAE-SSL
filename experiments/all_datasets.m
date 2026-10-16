%% Run PSAE-SSL on all benchmark datasets
% Paper protocol: 10 independent seeds for main tables (Section 4.4).
% Results saved per dataset; already-finished datasets are skipped on re-run.

clear; clc; close all;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
dataRoot   = fullfile(projectRoot, 'data');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

%% ---- Protocol (aligned with paper Section 4.4 / Tables 2–3) ----
usr       = get_user_settings();   % all settings: src/utils/get_user_settings.m
require_settings_confirmed();  % refuse to run until settings reviewed
numSeeds  = usr.numSeeds;          % independent seeds per dataset
mainRuns  = usr.mainRuns;          % seeds entering the main tables
labelFrac = usr.labelFrac;
datasets  = usr.datasets;
ensure_datasets(projectRoot, datasets, usr);   % downloads from Zenodo if missing
metricNames = {'NMI', 'Q', 'Precision', 'Recall', 'F1'};

summary = struct();
for di = 1:numel(datasets)
    name = datasets{di};
    outFile = fullfile(resultsDir, ['psae_ssl_' name '.mat']);
    if result_is_current(outFile, numSeeds)
        fprintf('[%s] already done, skipping\n', name);
        S = load(outFile, 'meanScores', 'stdScores');
        summary.(name) = S;
        continue
    end

    folder = fullfile(dataRoot, name);
    if ~exist(folder, 'dir')
        warning('[%s] data folder not found (%s), skipping.', name, folder);
        continue
    end

    cfg = get_dataset_config(name);
    try
        [adjList, allLabels, K] = load_dataset(name, dataRoot);
    catch ME
        warning('[%s] load_dataset failed: %s — skipping.', name, ME.message);
        continue
    end
    n = numel(adjList);
    if isempty(allLabels)
        warning('[%s] no ground-truth labels found, skipping.', name);
        continue
    end
    fprintf('\n=== %s: n=%d, K=%d, layers=%s, batch=%d ===\n', ...
        name, n, K, mat2str(cfg.layers), cfg.batchSize);

    scores = nan(numSeeds, numel(metricNames));
    bestKs = nan(numSeeds, 1);
    trainTime = nan(numSeeds, 1);
    inferTime = nan(numSeeds, 1);
    preds = cell(numSeeds, 1);
    labeledSets = cell(numSeeds, 1);
    nFail = 0;

    for s = 1:numSeeds
        try
            r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                struct('labelFrac', labelFrac, 'stratified', false, 'B', cfg.B));
            scores(s, :) = [r.NMI, r.Q, r.Precision, r.Recall, r.F1];
            bestKs(s) = r.bestK;
            trainTime(s) = r.trainTime;
            inferTime(s) = r.inferTime;
            preds{s} = r.predLabels;
            labeledSets{s} = r.labeledIdx;
            fprintf('[%s] seed=%d NMI=%.4f Q=%.4f P=%.4f R=%.4f F1=%.4f (K=%d, %.1fs)\n', ...
                name, s, scores(s, :), r.bestK, r.trainTime + r.inferTime);
        catch ME
            nFail = nFail + 1;
            warning('[%s] seed %d failed: %s', name, s, ME.message);
        end
    end

    nOk = sum(any(~isnan(scores), 2));
    fprintf('[%s] finished: %d/%d seeds ok, %d failed\n', name, nOk, numSeeds, nFail);

    if nOk == 0
        warning('[%s] all seeds failed — result file NOT saved.', name);
        continue
    end

    meanScores = mean(scores(1:min(mainRuns, size(scores,1)), :), 1, 'omitnan');
    stdScores  = std(scores(1:min(mainRuns, size(scores,1)), :), 0, 1, 'omitnan');

    matlabVersion = version; %#ok<NASGU>
    save(outFile, 'name', 'metricNames', 'scores', 'meanScores', 'stdScores', ...
        'bestKs', 'trainTime', 'inferTime', 'preds', 'labeledSets', ...
        'cfg', 'numSeeds', 'mainRuns', 'labelFrac', 'matlabVersion', 'nFail', 'nOk');
    summary.(name) = struct('meanScores', meanScores, 'stdScores', stdScores, ...
        'nOk', nOk, 'nFail', nFail);
end

save(fullfile(resultsDir, 'psae_ssl_all.mat'), 'summary', 'metricNames');
fprintf('\nDone. Summary saved to results/psae_ssl_all.mat\n');
