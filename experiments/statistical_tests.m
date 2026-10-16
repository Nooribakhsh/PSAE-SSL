%% STATISTICAL_TESTS  Friedman (all 5 metrics) + Wilcoxon (NMI)
%  Paper Section 5.3.1:  Tables 22-26 (Friedman: NMI, Q, Precision, Recall, F1)
%                        Table 14      (Wilcoxon signed-rank on NMI)
%
%  Run AFTER all_datasets (or nmi_q_log).
%
%  TWO MODES (printed at the start)
%  --------------------------------
%  A) PER-RUN mode  - exactly the paper's protocol: per dataset, the 10 baselines
%     and PSAE-SSL are ranked on every run and the ranks are averaged
%     (Tables 22-26); Wilcoxon is applied per dataset over the runs (Table 14).
%     Needs per-run baseline scores in  results/baseline_runs.mat  with
%        runs        [nRuns x 10 datasets x 10 baselines x 5 metrics]
%        methodNames {'DANMF','CDDTA',...}   (baseline order of the paper)
%     (dataset order: karate,dolphins,football,polbooks,polblogs,email,cora,
%      citeseer,pubmed,dblp ; metric order: NMI,Q,Precision,Recall,F1)
%  B) MEAN mode (default, no baseline code in this package) - baseline MEANS
%     are the values printed in the manuscript (Tables 9-13, see
%     src/utils/paper_baseline_means.m); PSAE-SSL means are measured here
%     (paper values fill datasets that have not been run). Friedman then ranks
%     the 11 methods over the 10 dataset means, Wilcoxon pairs PSAE-SSL with
%     each baseline over the 10 datasets. This is NOT the per-dataset
%     structure of Tables 14 / 22-26 - report it as a supplementary test.
%
%  PSAE-SSL: usr.statSeeds = 30 runs per dataset (paper Section 5.3.1), run here and
%  cached in results/stat_runs/ (seeds 1-10 are reused from the main experiment).
%  Outputs: results/statistical_tests.mat and results/tables/stat_*.csv

clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
tablesDir  = fullfile(resultsDir, 'tables');
if ~exist(tablesDir, 'dir'); mkdir(tablesDir); end
usr = get_user_settings();

datasets = {'karate','dolphins','football','polbooks','polblogs', ...
            'email','cora','citeseer','pubmed','dblp'};
dsDisp   = {'Karate','Dolphin','Football','Polbooks','Polblogs', ...
            'Email','Cora','CiteSeer','PubMed','DBLP'};
metricNames = {'NMI','Q','Precision','Recall','F1'};
metricKeys  = {'nmi','q','precision','recall','f1'};
nD = numel(datasets);

%% ---- PSAE-SSL runs: usr.statSeeds (= 30) seeds per dataset, resumable ----
% Seeds 1..10 are taken from the main experiment (same protocol, same seeds) when
% available; the remaining seeds are run here. Every finished seed is cached in
% results/stat_runs/<dataset>_s<k>.mat, so an interrupted run continues where it stopped.
statDir = fullfile(resultsDir, 'stat_runs');
if ~exist(statDir, 'dir'); mkdir(statDir); end
ensure_datasets(projectRoot, usr.statDatasets, usr);
for di = 1:nD
    name = datasets{di};
    if ~any(strcmp(usr.statDatasets, name)), continue; end
    cfg = get_dataset_config(name);
    loaded = false;
    for s = 1:usr.statSeeds
        sf = fullfile(statDir, sprintf('%s_s%d.mat', name, s));
        if exist(sf, 'file'), continue; end
        sc = reuse_main_run(resultsDir, name, s);
        if isempty(sc)
            if ~loaded
                [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
                loaded = true;
            end
            try
                r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                    struct('labelFrac', usr.labelFrac, 'stratified', false, 'B', cfg.B));
                sc = [r.NMI, r.Q, r.Precision, r.Recall, r.F1];
                fprintf('[%s] stat run %d/%d  NMI=%.4f\n', name, s, usr.statSeeds, sc(1));
            catch ME
                warning('[%s] seed %d failed: %s', name, s, ME.message);
                continue
            end
        end
        scores = sc; %#ok<NASGU>
        save(sf, 'scores');
    end
end
% collect everything that is available
psaeRuns = cell(nD, 1);
for di = 1:nD
    rows = [];
    for s = 1:usr.statSeeds
        sf = fullfile(statDir, sprintf('%s_s%d.mat', datasets{di}, s));
        if exist(sf, 'file'), L = load(sf, 'scores'); rows(end+1, :) = L.scores; end %#ok<SAGROW>
    end
    psaeRuns{di} = rows;
end
haveMeasured = ~cellfun(@isempty, psaeRuns);
fprintf('[stats] PSAE-SSL runs available for %d/%d datasets (%s runs each).\n', sum(haveMeasured), nD, ...
    mat2str(cellfun(@(c) size(c, 1), psaeRuns)'));

%% ---- choose mode ----
brf = fullfile(resultsDir, 'baseline_runs.mat');
perRun = exist(brf, 'file') == 2;
if perRun
    BR = load(brf);
    methodNames = [BR.methodNames(:)', {'PSAE-SSL'}];
    fprintf('[stats] MODE A (per-run, paper protocol) - baseline runs from %s\n', brf);
else
    P = paper_baseline_means();
    bmf = fullfile(resultsDir, 'baseline_means.mat');
    if exist(bmf, 'file'), M = load(bmf); P = M.B; fprintf('[stats] baseline means from %s\n', bmf); end
    methodNames = P.methods;
    fprintf('[stats] MODE B (means over datasets) - baseline means from the manuscript.\n');
    if ~all(haveMeasured)
        fprintf('[stats]   PSAE-SSL values missing for: %s -> manuscript values used there.\n', ...
            strjoin(dsDisp(~haveMeasured), ', '));
    end
end
nM = numel(methodNames);
psaeCol = nM;

result = struct('mode', ternary(perRun, 'per-run', 'means'), 'methodNames', {methodNames}, ...
                'datasets', {dsDisp}, 'metricNames', {metricNames});

if perRun
    %% ================= MODE A =================
    for mi = 1:5
        avgRankDS = nan(nD, nM);      % dataset x method (Tables 22-26 body)
        for di = 1:nD
            if isempty(psaeRuns{di}), continue; end
            nr = min(size(BR.runs, 1), size(psaeRuns{di}, 1));
            X = [squeeze(BR.runs(1:nr, di, :, mi)), psaeRuns{di}(1:nr, mi)];
            X = reshape(X, nr, nM);
            R = zeros(nr, nM);
            for r = 1:nr
                R(r, :) = (nM + 1) - avg_rank(X(r, :)');   % best = rank 1
            end
            avgRankDS(di, :) = mean(R, 1);
        end
        avgRank = mean(avgRankDS, 1, 'omitnan');
        [~, o] = sort(avgRank); overall = zeros(1, nM); overall(o) = 1:nM;
        result.friedman(mi) = struct('metric', metricNames{mi}, 'rankByDataset', avgRankDS, ...
            'avgRank', avgRank, 'overallRank', overall); %#ok<SAGROW>
        fprintf('[stats] Friedman %-9s  PSAE-SSL avg rank = %.3f (overall %d of %d)\n', ...
            metricNames{mi}, avgRank(psaeCol), overall(psaeCol), nM);
        write_rank_csv(fullfile(tablesDir, ['stat_friedman_' metricKeys{mi} '.csv']), ...
            dsDisp, methodNames, avgRankDS, avgRank, overall);
    end
    % Wilcoxon per dataset on NMI (Table 14)
    pRaw = nan(nM - 1, nD); pHolm = pRaw;
    for di = 1:nD
        if isempty(psaeRuns{di}), continue; end
        nr = min(size(BR.runs, 1), size(psaeRuns{di}, 1));
        for j = 1:nM - 1
            res = wilcoxon_signed_rank(psaeRuns{di}(1:nr, 1), squeeze(BR.runs(1:nr, di, j, 1)));
            pRaw(j, di) = res.p;
        end
        pHolm(:, di) = holm_adjust(pRaw(:, di)')';
    end
    result.wilcoxon = struct('pRaw', pRaw, 'pHolm', pHolm, 'methods', {methodNames(1:end-1)});
    write_matrix_csv(fullfile(tablesDir, 'stat_wilcoxon_nmi.csv'), methodNames(1:end-1), dsDisp, pRaw);
    fprintf('[stats] Wilcoxon per dataset (NMI) saved: results/tables/stat_wilcoxon_nmi.csv\n');
else
    %% ================= MODE B =================
    for mi = 1:5
        base = P.(metricKeys{mi});                  % methods x datasets (manuscript)
        X = base';                                  % datasets x methods
        for di = 1:nD
            if ~isempty(psaeRuns{di}), X(di, psaeCol) = mean(psaeRuns{di}(:, mi), 'omitnan'); end
        end
        fr = friedman_test(X, true);
        [~, o] = sort(fr.avgRanks); overall = zeros(1, nM); overall(o) = 1:nM;
        rankDS = zeros(nD, nM);
        for di = 1:nD, rankDS(di, :) = ((nM + 1) - avg_rank(X(di, :)'))'; end
        result.friedman(mi) = struct('metric', metricNames{mi}, 'chi2', fr.chi2, 'df', fr.df, ...
            'p', fr.p, 'FF', fr.FF, 'pFF', fr.pFF, 'avgRank', fr.avgRanks, ...
            'overallRank', overall, 'rankByDataset', rankDS, 'N', fr.N); %#ok<SAGROW>
        fprintf('[stats] Friedman %-9s chi2=%.3f p=%.3e | PSAE-SSL avg rank %.3f (overall %d of %d)\n', ...
            metricNames{mi}, fr.chi2, fr.p, fr.avgRanks(psaeCol), overall(psaeCol), nM);
        write_rank_csv(fullfile(tablesDir, ['stat_friedman_' metricKeys{mi} '.csv']), ...
            dsDisp, methodNames, rankDS, fr.avgRanks, overall);
    end
    % Wilcoxon over the 10 dataset means, NMI
    X = P.nmi';
    for di = 1:nD
        if ~isempty(psaeRuns{di}), X(di, psaeCol) = mean(psaeRuns{di}(:, 1), 'omitnan'); end
    end
    rawP = nan(1, nM - 1);
    for j = 1:nM - 1
        res = wilcoxon_signed_rank(X(:, psaeCol), X(:, j));
        w(j) = res; %#ok<SAGROW>
        rawP(j) = res.p;
    end
    pH = holm_adjust(rawP);
    fprintf('[stats] Wilcoxon over dataset means (NMI), Holm-adjusted:\n');
    for j = 1:nM - 1
        fprintf('  vs %-9s p=%.4e pHolm=%.4e (wins/ties/losses=%d/%d/%d)\n', methodNames{j}, ...
            rawP(j), pH(j), w(j).wins, w(j).ties, w(j).losses);
    end
    result.wilcoxon = struct('pRaw', rawP, 'pHolm', pH, 'methods', {methodNames(1:end-1)});
    result.nmiMatrix = X;
end

save(fullfile(resultsDir, 'statistical_tests.mat'), 'result');
fprintf('[stats] Saved %s\n', fullfile(resultsDir, 'statistical_tests.mat'));

%% ---- local helpers ----
function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function write_rank_csv(file, rowNames, colNames, M, avgRank, overall)
fid = fopen(file, 'w');
fprintf(fid, 'Dataset,%s\n', strjoin(colNames, ','));
for i = 1:size(M, 1)
    fprintf(fid, '%s', rowNames{i});
    fprintf(fid, ',%.2f', M(i, :)); fprintf(fid, '\n');
end
fprintf(fid, 'Avg. Rank'); fprintf(fid, ',%.2f', avgRank); fprintf(fid, '\n');
fprintf(fid, 'Overall Rank'); fprintf(fid, ',%d', overall); fprintf(fid, '\n');
fclose(fid);
end

function write_matrix_csv(file, rowNames, colNames, M)
fid = fopen(file, 'w');
fprintf(fid, 'Method,%s\n', strjoin(colNames, ','));
for i = 1:size(M, 1)
    fprintf(fid, '%s', rowNames{i});
    fprintf(fid, ',%.3e', M(i, :)); fprintf(fid, '\n');
end
fclose(fid);
end

function sc = reuse_main_run(resultsDir, name, s)
% Row s of the main experiment (same protocol and seed) if it exists, else [].
sc = [];
cands = {['psae_ssl_' name '.mat'], ['nmi_q_' name '.mat']};
for c = 1:numel(cands)
    f = fullfile(resultsDir, cands{c});
    if exist(f, 'file')
        L = load(f, 'scores');
        if isfield(L, 'scores') && size(L.scores, 1) >= s && size(L.scores, 2) >= 5 ...
                && ~any(isnan(L.scores(s, 1:5)))
            sc = L.scores(s, 1:5); return
        end
    end
end
end
