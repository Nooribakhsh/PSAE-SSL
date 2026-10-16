%% Ablation study (6 variants) on all datasets, numSeeds independent runs.
% Output: results/ablation_<dataset>_<tag>.mat per (dataset, variant),
%         results/ablation_all.mat (mean/std tensors).
% Resumable: finished (dataset, variant) pairs are skipped.
% Put in experiments/. Needs psae_pipeline.m in src/utils.

clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

usr = get_user_settings();   % settings: src/utils/get_user_settings.m
datasets = usr.datasets;
ensure_datasets(projectRoot, datasets, usr);
variants = {'full','-Node2Vec','SAE','-sep','-SSL','k-means'};
tags     = {'full','noN2V','SAE','nosep','noSSL','kmeans'};
numSeeds  = usr.numSeeds;   % same as all_datasets.m
labelFrac = usr.labelFrac;
% -Node2Vec: sparse adj + projection (no n-cap; see adjlist_to_features)
metricNames = {'NMI','Q','Precision','Recall','F1'};

for di = 1:numel(datasets)
    name = datasets{di};
    cfg = get_dataset_config(name);
    [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
    n = numel(adjList);
    if isempty(allLabels)
        warning('[%s] no labels, skipping.', name); continue
    end

    % Node2Vec embeddings are shared by all variants except '-Node2Vec'
    % (same seed -> same Z0 as in all_datasets). Cached per seed.
    Z0cache = cell(numSeeds, 1);

    for vi = 1:numel(variants)
        variant = variants{vi};
        outFile = fullfile(resultsDir, ['ablation_' name '_' tags{vi} '.mat']);
        if result_is_current(outFile, numSeeds)
            fprintf('[%s|%s] done, skipping\n', name, variant); continue
        end

        scores = nan(numSeeds, numel(metricNames));
        sepScores = nan(numSeeds, 2); % FDR, SC (E5 / Table 8)
        bestKs = nan(numSeeds, 1);
        errors = cell(numSeeds, 1);
        skipped = false;
        for s = 1:numSeeds
            try
                opts = struct('variant', variant, 'labelFrac', labelFrac, ...
                    'stratified', false, 'computeSep', true);
                if ~strcmp(variant, '-Node2Vec')
                    if isempty(Z0cache{s})
                        n2v = get_n2v_options(cfg, n, s);
                        Z0cache{s} = node2vec(adjList, n2v);
                    end
                    opts.Z0 = Z0cache{s};
                end
                r = psae_pipeline(adjList, allLabels, K, cfg, s, opts);
                scores(s, :) = [r.NMI, r.Q, r.Precision, r.Recall, r.F1];
                bestKs(s) = r.bestK;
                if isfield(r, 'sepAfter') && isstruct(r.sepAfter)
                    sepScores(s, 1) = r.sepAfter.FDR;
                    sepScores(s, 2) = r.sepAfter.SC;
                end
                fprintf('[%s|%s] seed=%d NMI=%.4f Q=%.4f F1=%.4f\n', ...
                    name, variant, s, scores(s,1), scores(s,2), scores(s,5));
            catch ME
                errors{s} = ME.message;
                warning('[%s|%s] seed %d failed: %s', name, variant, s, ME.message);
            end
        end
        meanScores = mean(scores, 1, 'omitnan');
        stdScores  = std(scores, 0, 1, 'omitnan');
        matlabVersion = version; %#ok<NASGU>
        meanSep = mean(sepScores, 1, 'omitnan');
        stdSep  = std(sepScores, 0, 1, 'omitnan');
        save(outFile, 'name', 'variant', 'metricNames', 'scores', 'meanScores', ...
            'stdScores', 'sepScores', 'meanSep', 'stdSep', 'bestKs', 'errors', 'skipped', 'numSeeds', 'labelFrac', ...
            'cfg', 'matlabVersion');
    end
end

%% ---- collect: rawAbl [seed x metric x variant x dataset] ----
rawAbl = nan(numSeeds, 5, numel(variants), numel(datasets));
for di = 1:numel(datasets)
    for vi = 1:numel(variants)
        f = fullfile(resultsDir, ['ablation_' datasets{di} '_' tags{vi} '.mat']);
        if exist(f, 'file')
            S = load(f, 'scores');
            k = min(numSeeds, size(S.scores, 1));
            rawAbl(1:k, :, vi, di) = S.scores(1:k, :);
        end
    end
end
meanAbl = permute(squeeze(mean(rawAbl, 1, 'omitnan')), [2 3 1]);  % variant x dataset x metric
stdAbl  = permute(squeeze(std(rawAbl, 0, 1, 'omitnan')), [2 3 1]);
save(fullfile(resultsDir, 'ablation_all.mat'), 'rawAbl', 'meanAbl', 'stdAbl', ...
    'variants', 'datasets', 'metricNames', 'numSeeds');
