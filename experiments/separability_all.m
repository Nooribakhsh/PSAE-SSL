%% Latent-space separability (FDR, SC, DBI, CHI): Node2Vec vs PSAE-SSL, all datasets.
% Output: results/separability_<dataset>.mat, results/separability_all.mat
% Both representations are scored against the partition AHC finds on them
% (same protocol as separability_karate.m). Relative improvement is
% (after-before)/before for FDR/SC/CHI and (before-after)/before for DBI
% (lower is better). NMI gain = NMI(PSAE-SSL) - NMI(Node2Vec+AHC).
% Put in experiments/. Needs psae_pipeline.m (computeSep option).

clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

usr = get_user_settings();   % settings: src/utils/get_user_settings.m
datasets = usr.datasets;
ensure_datasets(projectRoot, datasets, usr);
numSeeds = usr.numSeeds; labelFrac = usr.labelFrac;
fn = {'FDR','SC','DBI','CHI'};

for di = 1:numel(datasets)
    name = datasets{di};
    outFile = fullfile(resultsDir, ['separability_' name '.mat']);
    if result_is_current(outFile, numSeeds), fprintf('[%s] done, skipping\n', name); continue; end
    cfg = get_dataset_config(name);
    [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
    if isempty(allLabels), warning('[%s] no labels, skipping', name); continue; end

    before = nan(numSeeds, 4); after = nan(numSeeds, 4); nmiAfter = nan(numSeeds, 1);
    errors = cell(numSeeds, 1);
    for s = 1:numSeeds
        try
            r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                struct('labelFrac', labelFrac, 'stratified', false, ...
                       'computeSep', true));
            for j = 1:4
                before(s, j) = r.sepBefore.(fn{j});
                after(s, j)  = r.sepAfter.(fn{j});
            end
            nmiAfter(s) = r.NMI;
            fprintf('[%s] seed=%d FDR %.2f->%.2f  SC %.2f->%.2f  DBI %.2f->%.2f  CHI %.0f->%.0f\n', ...
                name, s, before(s,1), after(s,1), before(s,2), after(s,2), ...
                before(s,3), after(s,3), before(s,4), after(s,4));
        catch ME
            errors{s} = ME.message;
            warning('[%s] seed %d failed: %s', name, s, ME.message);
        end
    end
    meanBefore = mean(before, 1, 'omitnan'); meanAfter = mean(after, 1, 'omitnan');
    stdBefore = std(before, 0, 1, 'omitnan'); stdAfter = std(after, 0, 1, 'omitnan');
    relImp = (meanAfter - meanBefore) ./ abs(meanBefore) * 100;
    relImp(3) = (meanBefore(3) - meanAfter(3)) / abs(meanBefore(3)) * 100;   % DBI
    matlabVersion = version; %#ok<NASGU>
    save(outFile, 'name', 'fn', 'before', 'after', 'nmiAfter', 'meanBefore', ...
        'meanAfter', 'stdBefore', 'stdAfter', 'relImp', 'errors', 'numSeeds', ...
        'labelFrac', 'matlabVersion');
end

%% collect
mB = nan(numel(datasets), 4); mA = mB; rI = mB; sB = mB; sA = mB;
for di = 1:numel(datasets)
    f = fullfile(resultsDir, ['separability_' datasets{di} '.mat']);
    if exist(f, 'file')
        S = load(f);
        mB(di,:) = S.meanBefore; mA(di,:) = S.meanAfter; rI(di,:) = S.relImp;
        sB(di,:) = S.stdBefore;  sA(di,:) = S.stdAfter;
    end
end
save(fullfile(resultsDir, 'separability_all.mat'), 'datasets', 'fn', 'mB', 'mA', 'sB', 'sA', 'rI');
