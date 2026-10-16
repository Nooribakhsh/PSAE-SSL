%% Sensitivity to the number of branches B = 1..5 on all datasets.
% Output: results/branches_<dataset>.mat, results/branches_all.mat
%   nmiB  [numSeeds x 5(B)]  per dataset ; meanB/stdB [nDatasets x 5]
% Resumable per dataset. Put in experiments/.

clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

usr = get_user_settings();   % settings: src/utils/get_user_settings.m
datasets = usr.datasets;
ensure_datasets(projectRoot, datasets, usr);
Bvalues = usr.branchValues; numSeeds = usr.numSeeds; labelFrac = usr.labelFrac;

for di = 1:numel(datasets)
    name = datasets{di};
    outFile = fullfile(resultsDir, ['branches_' name '.mat']);
    if result_is_current(outFile, numSeeds), fprintf('[%s] done, skipping\n', name); continue; end
    cfg = get_dataset_config(name);
    [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
    if isempty(allLabels), warning('[%s] no labels, skipping', name); continue; end

    nmiB = nan(numSeeds, numel(Bvalues)); errors = cell(numSeeds, numel(Bvalues));
    for s = 1:numSeeds
        n2v = get_n2v_options(cfg, numel(adjList), s);
        Z0 = node2vec(adjList, n2v);      % shared by all B for this seed
        for bi = 1:numel(Bvalues)
            try
                r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                    struct('B', Bvalues(bi), 'labelFrac', labelFrac, ...
                           'stratified', false, 'Z0', Z0));
                nmiB(s, bi) = r.NMI;
                fprintf('[%s] seed=%d B=%d NMI=%.4f\n', name, s, Bvalues(bi), r.NMI);
            catch ME
                errors{s, bi} = ME.message;
                warning('[%s] seed %d B=%d failed: %s', name, s, Bvalues(bi), ME.message);
            end
        end
    end
    meanNMI = mean(nmiB, 1, 'omitnan'); stdNMI = std(nmiB, 0, 1, 'omitnan');
    matlabVersion = version; %#ok<NASGU>
    save(outFile, 'name', 'Bvalues', 'nmiB', 'meanNMI', 'stdNMI', 'errors', ...
        'numSeeds', 'labelFrac', 'cfg', 'matlabVersion');
end

%% collect
meanB = nan(numel(datasets), numel(Bvalues)); stdB = meanB;
for di = 1:numel(datasets)
    f = fullfile(resultsDir, ['branches_' datasets{di} '.mat']);
    if exist(f, 'file'), S = load(f); meanB(di, :) = S.meanNMI; stdB(di, :) = S.stdNMI; end
end
save(fullfile(resultsDir, 'branches_all.mat'), 'datasets', 'Bvalues', 'meanB', 'stdB');
