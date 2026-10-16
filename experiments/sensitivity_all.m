%% sensitivity_all
% Runs ALL sensitivity experiments sequentially:
%   1) Number of branches B = 1..5   (sensitivity_branches_all)
%   2) Label / supervision ratios    (sensitivity_label_all)
%
% Usage (from experiments/ folder or any path after addpath):
%   sensitivity_all
%
% Results:
%   results/branches_*.mat  +  results/branches_all.mat
%   results/sens_label_*.mat + results/sens_label_all.mat

clear; clc; close all;

fprintf('\n============================================================\n');
fprintf('  PSAE-SSL  |  SENSITIVITY EXPERIMENTS (ALL)\n');
fprintf('============================================================\n\n');

fprintf('>>> [1/2] Sensitivity to number of branches (B = 1..5)\n');
fprintf('------------------------------------------------------------\n');
sensitivity_branches_all;

fprintf('\n>>> [2/2] Sensitivity to label / supervision ratio\n');
fprintf('------------------------------------------------------------\n');
sensitivity_label_all;

fprintf('\n============================================================\n');
fprintf('  All sensitivity experiments finished.\n');
fprintf('============================================================\n');
