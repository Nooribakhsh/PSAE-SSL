%% Collect PSAE-SSL results into one tensor (proposed-method-only package).
% Reads results/psae_ssl_<dataset>.mat produced by all_datasets.
% Output: results/collected.mat
%   methods = {'PSAE-SSL'}
%   raw     [numSeeds x 5 metrics x 1 x nDatasets]
%   meanT, stdT  [1 x nDatasets x 5]

clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);

usr = get_user_settings();
datasets = usr.datasets;
methods  = {'PSAE-SSL'};
metricNames = {'NMI','Q','Precision','Recall','F1'};
numSeeds = usr.numSeeds;
mainRuns = usr.mainRuns;

raw = nan(numSeeds, 5, numel(methods), numel(datasets));
for di = 1:numel(datasets)
    f = fullfile(resultsDir, ['psae_ssl_' datasets{di} '.mat']);
    if ~exist(f, 'file')
        % also accept nmi_q_<name>.mat from nmi_q_log
        f2 = fullfile(resultsDir, ['nmi_q_' datasets{di} '.mat']);
        if exist(f2, 'file')
            S = load(f2, 'scores');
            k = min(numSeeds, size(S.scores, 1));
            raw(1:k, :, 1, di) = S.scores(1:k, :);
            fprintf('[%s] loaded from nmi_q_*.mat (%d seeds)\n', datasets{di}, k);
        else
            fprintf('[%s] missing — skip\n', datasets{di});
        end
        continue
    end
    S = load(f);
    if isfield(S, 'scores')
        k = min(numSeeds, size(S.scores, 1));
        raw(1:k, :, 1, di) = S.scores(1:k, :);
    elseif isfield(S, 'meanScores')
        % fallback: only means available
        raw(1, :, 1, di) = S.meanScores(:)';
    end
    fprintf('[%s] loaded (%d seeds)\n', datasets{di}, sum(any(~isnan(raw(:,:,1,di)), 2)));
end

rawMain = raw(1:mainRuns, :, :, :);
meanT = squeeze(mean(rawMain, 1, 'omitnan'));   % may be 5 x nD or nD x 5
stdT  = squeeze(std(rawMain, 0, 1, 'omitnan'));
if isvector(meanT)
    meanT = reshape(meanT, [1 numel(datasets) 5]);
    stdT  = reshape(stdT,  [1 numel(datasets) 5]);
elseif size(meanT, 1) == 5
    meanT = permute(meanT, [3 2 1]);  % 1 x nD x 5
    stdT  = permute(stdT,  [3 2 1]);
else
    meanT = permute(meanT, [2 3 1]);
    stdT  = permute(stdT,  [2 3 1]);
end

save(fullfile(resultsDir, 'collected.mat'), 'raw', 'meanT', 'stdT', ...
    'methods', 'datasets', 'metricNames', 'numSeeds', 'mainRuns');
fprintf('Saved results/collected.mat\n');

fprintf('\n%-12s %8s %8s %8s %8s %8s\n', 'Dataset','NMI','Q','P','R','F1');
for di = 1:numel(datasets)
end
