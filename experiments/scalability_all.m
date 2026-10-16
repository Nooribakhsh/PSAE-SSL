%% Scalability of PSAE-SSL (train time, inference time, peak memory).
% Runs ALL datasets listed in usr.datasets (src/utils/get_user_settings.m) one after
% another, usr.numSeeds seeds each (fresh MATLAB process per run, Windows-safe).
% Missing datasets are downloaded from Zenodo first (usr.autoDownload).
% Result: results/scalability_all.mat  (+ per-seed cache files, so an interrupted run
% simply continues when you press Run again).
% To measure only some datasets, edit usr.datasets in get_user_settings.m.
clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
usr = get_user_settings();
scalability_core(usr.datasets, 'scalability_all');
