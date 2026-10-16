%% Ablation: how much does Stage 2 (PSAE) actually contribute?  (resumable)
% Modes (all share the SAME Node2Vec embedding, labeled split, and Stage 3):
%   'none' : no PSAE - Stage 3 + label-guided refinement on Node2Vec Z0
%   'v6'   : PSAE with the original fine-tuning
%   'v7'   : PSAE with the stabilised fine-tuning + embedding selection
% Report the numbers exactly as measured. Do NOT pick a mode per dataset
% by looking at NMI on all nodes.
%
% Output: results/ablation_psae_mode.mat, results/ablation_psae_mode.csv
%         (per-run cache in results/cache_abl/)

clear; clc; close all;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
cacheDir   = fullfile(resultsDir, 'cache_abl');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end
if ~exist(cacheDir, 'dir');   mkdir(cacheDir);   end

% ---- settings (edit here) ----
datasets = {'karate','citeseer'};     % add 'pubmed' when time allows
usr      = get_user_settings();        % settings: src/utils/get_user_settings.m
ratios   = usr.labelRatios;
modes    = {'none','v6','v7'};
numSeeds = usr.numSeeds;
ensure_datasets(projectRoot, datasets, usr);
% ------------------------------

nD = numel(datasets); nR = numel(ratios); nM = numel(modes);
NMI = nan(nD, nR, nM, numSeeds);

for di = 1:nD
    name = datasets{di};
    folder = fullfile(projectRoot, 'data', name);
    if ~exist(folder, 'dir'), warning('[%s] data missing - skip', name); continue; end
    cfg = get_dataset_config(name);
    [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));

    for s = 1:numSeeds
        z0File = fullfile(cacheDir, sprintf('z0_%s_s%d.mat', name, s));
        if exist(z0File, 'file')
            Z = load(z0File, 'Z0'); Z0 = Z.Z0;
        else
            n2v = get_n2v_options(cfg, numel(adjList), s);
            Z0 = node2vec(adjList, n2v);
            save(z0File, 'Z0', '-v7.3');
        end
        for ri = 1:nR
            for mi = 1:nM
                rf = fullfile(cacheDir, sprintf('%s_s%d_r%02d_%s.mat', name, s, round(100*ratios(ri)), modes{mi}));
                if exist(rf, 'file')
                    R = load(rf, 'nmiVal'); NMI(di, ri, mi, s) = R.nmiVal; continue
                end
                try
                    t0 = tic;
                    r = psae_pipeline(adjList, allLabels, K, cfg, s, struct( ...
                        'labelFrac', ratios(ri), 'stratified', true, 'Z0', Z0, ...
                        'forceTargetK', true, 'labelGuided', true, 'psaeMode', modes{mi}));
                    nmiVal = r.NMI; elapsed = toc(t0);
                    save(rf, 'nmiVal', 'elapsed');
                    NMI(di, ri, mi, s) = nmiVal;
                    fprintf('[%s] seed=%d ratio=%.0f%% mode=%s NMI=%.4f (%.0fs)\n', ...
                        name, s, 100*ratios(ri), modes{mi}, nmiVal, elapsed);
                catch ME
                    warning('[%s] seed=%d ratio=%.0f%% mode=%s failed: %s', ...
                        name, s, 100*ratios(ri), modes{mi}, ME.message);
                end
            end
        end
    end
end

% ---- summary table (mean +- std over seeds) ----
fid = fopen(fullfile(resultsDir, 'ablation_psae_mode.csv'), 'w');
fprintf(fid, 'dataset,mode,ratio,mean_nmi,std_nmi,n_runs\n');
for di = 1:nD
    fprintf('\n== %s ==\n%-6s', datasets{di}, 'mode');
    fprintf('%14.0f%%', 100*ratios); fprintf('\n');
    for mi = 1:nM
        fprintf('%-6s', modes{mi});
        for ri = 1:nR
            v = squeeze(NMI(di, ri, mi, :));
            m = mean(v, 'omitnan'); sd = std(v, 'omitnan'); nr = sum(~isnan(v));
            fprintf('  %.3f+-%.3f', m, sd);
            fprintf(fid, '%s,%s,%.2f,%.4f,%.4f,%d\n', datasets{di}, modes{mi}, ratios(ri), m, sd, nr);
        end
        fprintf('\n');
    end
end
fclose(fid);
save(fullfile(resultsDir, 'ablation_psae_mode.mat'), 'datasets', 'ratios', 'modes', 'NMI');
fprintf('\nSaved results/ablation_psae_mode.mat and .csv\n');
