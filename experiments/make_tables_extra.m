%% Extra tables from ablation / branches / separability / scalability (if present).
% Writes text summaries under results/tables/. Safe to run even if some
% experiment outputs are missing.

clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
outDir = fullfile(resultsDir, 'tables');
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = {'ablation_all.mat','branches_all.mat','separability_all.mat', ...
         'scalability_all.mat','sens_label_all.mat'};
for i = 1:numel(files)
    f = fullfile(resultsDir, files{i});
    if exist(f, 'file')
        S = load(f);
        out = fullfile(outDir, [files{i}(1:end-4) '_summary.mat']);
        save(out, '-struct', 'S');
        fprintf('Copied summary fields from %s\n', files{i});
    else
        fprintf('Skip (missing): %s\n', files{i});
    end
end
fprintf('Done.\n');
