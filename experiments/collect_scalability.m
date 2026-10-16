%% Print the scalability table (mean +/- std over seeds) from results/scalability_all.mat.
clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
order = {'karate','dolphins','football','polbooks','polblogs','email','cora','citeseer','pubmed','dblp'};
f = fullfile(resultsDir, 'scalability_all.mat');
if ~exist(f, 'file'), error('%s not found - run scalability_all first.', f); end
L = load(f, 'summary', 'numRuns');
S = L.summary;
fprintf('(mean +/- std over %d seeds)\n', L.numRuns);
fprintf('%-10s %22s %22s %20s\n', 'Dataset', 'Train (s)', 'Infer (s)', 'Peak (MB)');
for i = 1:numel(order)
    if ~isfield(S, order{i}), fprintf('%-10s   (not measured yet)\n', order{i}); continue; end
    q = S.(order{i});
    fprintf('%-10s %10.2f +/- %-8.2f %9.3f +/- %-7.3f %7.0f +/- %-5.0f\n', order{i}, ...
        q.train_mean, q.train_std, q.infer_mean, q.infer_std, q.peak_mean, q.peak_std);
end
