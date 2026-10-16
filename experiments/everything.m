%% everything
% Master script that executes the COMPLETE recommended experiment order
% for the PSAE-SSL package (proposed-method-only).
%
% Order (matches experiments/README_ORDER.txt):
%   1. all_datasets
%   2. ablation_all
%   3. sensitivity_all          (branches + label)
%   4. separability_all
%   5. scalability_all
%   6. collect_results
%   7. statistical_tests
%   8. make_tables_main
%   9. make_tables_rel
%
% All settings (datasets, seeds, epochs, ...): src/utils/get_user_settings.m
% Missing datasets are downloaded from Zenodo automatically.
%
% Notes
% - DBLP is large; expect long runtime and high memory.
% - Each individual script is resumable (already-finished parts are skipped).
% - You can interrupt and re-run this master script; it will continue.
% - For a quick smoke-test, uncomment the subset lines inside the
%   individual scripts first.
%
% Usage:
%   cd experiments
%   everything

clear; clc; close all;

setappdata(0, 'psae_t0', tic);   % NOTE: the scripts below call 'clear', which would erase a local timer
fprintf('\n################################################################\n');
fprintf('#  PSAE-SSL  |  FULL EXPERIMENT PIPELINE (everything)      #\n');
fprintf('################################################################\n\n');

%% 1. Main metrics on all datasets
fprintf('>>> [1/9] all_datasets  (main metrics, Tables 9-13 style)\n');
fprintf('------------------------------------------------------------\n');
all_datasets;

%% 2. Ablation study
fprintf('\n>>> [2/9] ablation_all\n');
fprintf('------------------------------------------------------------\n');
ablation_all;

%% 3. Sensitivity experiments (branches + label)
fprintf('\n>>> [3/9] sensitivity_all  (branches + label ratios)\n');
fprintf('------------------------------------------------------------\n');
sensitivity_all;

%% 4. Separability metrics
fprintf('\n>>> [4/9] separability_all\n');
fprintf('------------------------------------------------------------\n');
separability_all;

%% 5. Scalability (train/infer time + peak memory)
fprintf('\n>>> [5/9] scalability_all\n');
fprintf('------------------------------------------------------------\n');
scalability_all;

%% 6. Collect main results
fprintf('\n>>> [6/9] collect_results\n');
fprintf('------------------------------------------------------------\n');
collect_results;

%% 7. Statistical tests (Friedman + Wilcoxon)
fprintf('\n>>> [7/9] statistical_tests\n');
fprintf('------------------------------------------------------------\n');
statistical_tests;

%% 8. Main tables
fprintf('\n>>> [8/9] make_tables_main\n');
fprintf('------------------------------------------------------------\n');
make_tables_main;

%% 9. Tables derived from saved results (17-21 PSAE-SSL rows, 5, 28, 29)
fprintf('\n>>> [9/9] make_tables_rel\n');
fprintf('------------------------------------------------------------\n');
make_tables_rel;

%% Done
elapsed = toc(getappdata(0, 'psae_t0'));
fprintf('\n################################################################\n');
fprintf('#  ALL EXPERIMENTS FINISHED\n');
fprintf('#  Total wall-clock time: %.1f minutes (%.2f hours)\n', elapsed/60, elapsed/3600);
fprintf('################################################################\n');
fprintf('Results are under:  results/\n');
