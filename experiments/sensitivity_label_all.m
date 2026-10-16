%% Supervision-ratio sensitivity for PSAE-SSL only (resumable version).
% Datasets: karate, citeseer, pubmed.
% Ratios: 5%, 10%, 20%, 30%, 50% with stratified labeling.
%
% Every (dataset, seed) Node2Vec embedding and every (dataset, seed, ratio)
% run is cached in results/cache_sens/. If MATLAB stops or crashes, run the
% script again and it continues from the last finished run.
%
% Output: results/sens_label_<dataset>.mat, results/sens_label_all.mat
%         (partial results: results/sens_label_<dataset>_partial.mat)

clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
cacheDir   = fullfile(resultsDir, 'cache_sens');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end
if ~exist(cacheDir, 'dir');   mkdir(cacheDir);   end

% ---- settings (edit here) ----
usr      = get_user_settings();        % settings: src/utils/get_user_settings.m
datasets = usr.labelSensDatasets;
ratios   = usr.labelRatios;
numSeeds = usr.numSeeds;
useLabelGuided = true;   % label-guided Stage 3 refinement (see label_guided_clustering.m)
% ------------------------------

ensure_datasets(projectRoot, datasets, usr);
allNMI = nan(numel(datasets), numel(ratios), numSeeds);

for di = 1:numel(datasets)
    name = datasets{di};
    outFile = fullfile(resultsDir, ['sens_label_' name '.mat']);
    if result_is_current(outFile, numSeeds)
        S = load(outFile, 'nmi');
        allNMI(di, :, :) = permute(S.nmi, [2 1]);
        fprintf('[%s] done, skipping\n', name); continue
    end
    folder = fullfile(projectRoot, 'data', name);
    if ~exist(folder, 'dir')
        warning('[%s] data missing - skip', name); continue
    end
    cfg = get_dataset_config(name);
    [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
    nmi = nan(numSeeds, numel(ratios));

    for s = 1:numSeeds
        % ---- Node2Vec embedding (cached per dataset and seed) ----
        z0File = fullfile(cacheDir, sprintf('z0_%s_s%d.mat', name, s));
        Z0 = [];
        % skip Stage 1 if every ratio of this seed is already cached
        needZ0 = false;
        for ri = 1:numel(ratios)
            if ~exist(seed_file(cacheDir, name, s, ratios(ri)), 'file')
                needZ0 = true; break
            end
        end
        if needZ0
            if exist(z0File, 'file')
                Z = load(z0File, 'Z0'); Z0 = Z.Z0;
            else
                n2v = get_n2v_options(cfg, numel(adjList), s);
                Z0 = node2vec(adjList, n2v);
                save(z0File, 'Z0', '-v7.3');
            end
        end

        for ri = 1:numel(ratios)
            rf = seed_file(cacheDir, name, s, ratios(ri));
            if exist(rf, 'file')
                R = load(rf, 'nmiVal');
                nmi(s, ri) = R.nmiVal;
                fprintf('[%s] seed=%d ratio=%.0f%% cached\n', ...
                    name, s, 100*ratios(ri));
                continue
            end
            try
                t0 = tic;
                r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                    struct('labelFrac', ratios(ri), 'stratified', true, ...
                           'Z0', Z0, 'forceTargetK', true, ...
                           'labelGuided', useLabelGuided));
                nmiVal = r.NMI;
                elapsed = toc(t0);
                save(rf, 'nmiVal', 'elapsed');
                nmi(s, ri) = nmiVal;
                fprintf('[%s] seed=%d ratio=%.0f%% (%.0fs)\n', ...
                    name, s, 100*ratios(ri), elapsed);
            catch ME
                warning('[%s] seed=%d ratio=%.0f%% failed: %s', ...
                    name, s, 100*ratios(ri), ME.message);
            end
        end
    end

    meanNMI = mean(nmi, 1, 'omitnan');
    stdNMI  = std(nmi, 0, 1, 'omitnan');
    deltaNMI = 100 * (meanNMI(end) - meanNMI(1)) / max(meanNMI(1), eps);
    if any(isnan(nmi(:)))
        partialFile = fullfile(resultsDir, ['sens_label_' name '_partial.mat']);
        save(partialFile, 'name', 'ratios', 'nmi', 'meanNMI', 'stdNMI', 'deltaNMI', 'numSeeds');
        warning('[%s] %d runs missing - saved partial file only; run the script again.', ...
            name, sum(isnan(nmi(:))));
    else
        save(outFile, 'name', 'ratios', 'nmi', 'meanNMI', 'stdNMI', 'deltaNMI', 'numSeeds');
    end
    allNMI(di, :, :) = permute(nmi, [2 1]);
    fprintf('[%s] finished\n', name);
end

save(fullfile(resultsDir, 'sens_label_all.mat'), 'datasets', 'ratios', 'allNMI', 'numSeeds');
fprintf('Saved results/sens_label_all.mat\n');

function f = seed_file(cacheDir, name, s, ratio)
f = fullfile(cacheDir, sprintf('%s_s%d_r%02d_lg_v7.mat', name, s, round(100*ratio)));
end
