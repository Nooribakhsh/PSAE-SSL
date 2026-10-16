%% make_tables_rel  Paper tables that are derived from saved PSAE-SSL results.
%   Table 17-21 (PSAE-SSL row): max over runs of NMI, Q, Precision, Recall, F1
%   Table 29  : relative NMI improvement (%) of PSAE-SSL over the ablation variants
%   Table 28  : relative improvement (%) of latent separability (Node2Vec -> PSAE-SSL)
%   Table 5   : relative NMI change (%) between the smallest and largest label ratio
% Baseline rows of Tables 17-21 / 27 need baseline results and are NOT produced here.
% Each table is written to results/tables/*.csv; missing inputs are skipped.
clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
resultsDir = get_results_dir(projectRoot);
outDir = fullfile(resultsDir, 'tables');
if ~exist(outDir, 'dir'); mkdir(outDir); end
dsDisp = {'Karate','Dolphin','Football','Polbooks','Polblogs','Email','Cora','CiteSeer','PubMed','DBLP'};
dsKey  = {'karate','dolphins','football','polbooks','polblogs','email','cora','citeseer','pubmed','dblp'};

%% ---- Tables 17-21: max over runs (PSAE-SSL) ----
cf = fullfile(resultsDir, 'collected.mat');
if exist(cf, 'file')
    C = load(cf);
    mnames = {'NMI','Q','Precision','Recall','F1'};
    tnum   = [17 18 19 20 21];
    for m = 1:5
        fid = fopen(fullfile(outDir, sprintf('table%d_%s_max.csv', tnum(m), lower(mnames{m}))), 'w');
        fprintf(fid, 'Method,%s\n', strjoin(dsDisp, ','));
        fprintf(fid, 'PSAE-SSL');
        for di = 1:numel(dsKey)
            ci = find(strcmpi(C.datasets, dsKey{di}), 1);
            if isempty(ci), fprintf(fid, ',n/a'); continue; end
            v = C.raw(1:C.mainRuns, m, 1, ci);
            if all(isnan(v)), fprintf(fid, ',n/a'); else, fprintf(fid, ',%.4f', max(v)); end
        end
        fprintf(fid, '\n'); fclose(fid);
    end
    fprintf('Wrote Tables 17-21 (PSAE-SSL max rows).\n');
else
    fprintf('Skip Tables 17-21: %s missing (run collect_results).\n', cf);
end

%% ---- Table 29: relative improvement over ablation variants (NMI) ----
af = fullfile(resultsDir, 'ablation_all.mat');
if exist(af, 'file')
    A = load(af);                       % meanAbl: variant x dataset x metric
    vNames = A.variants;
    fid = fopen(fullfile(outDir, 'table29_rel_improvement_ablation.csv'), 'w');
    fprintf(fid, 'Dataset,%s\n', strjoin(vNames(2:end), ','));
    for di = 1:numel(dsKey)
        ci = find(strcmpi(A.datasets, dsKey{di}), 1);
        if isempty(ci), continue; end
        full = A.meanAbl(1, ci, 1);
        fprintf(fid, '%s', dsDisp{di});
        for vi = 2:numel(vNames)
            v = A.meanAbl(vi, ci, 1);
            fprintf(fid, ',%.2f', (full - v) / v * 100);
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('Wrote Table 29.\n');
else
    fprintf('Skip Table 29: %s missing (run ablation_all).\n', af);
end

%% ---- Table 28: separability improvement ----
sf = fullfile(resultsDir, 'separability_all.mat');
if exist(sf, 'file')
    Sp = load(sf);                      % rI: dataset x [FDR SC DBI CHI]
    fid = fopen(fullfile(outDir, 'table28_rel_improvement_separability.csv'), 'w');
    fprintf(fid, 'Dataset,FDR,SC,DBI,CHI\n');
    for di = 1:numel(dsKey)
        ci = find(strcmpi(Sp.datasets, dsKey{di}), 1);
        if isempty(ci), continue; end
        fprintf(fid, '%s,%.2f,%.2f,%.2f,%.2f\n', dsDisp{di}, Sp.rI(ci, :));
    end
    fclose(fid);
    fprintf('Wrote Table 28.\n');
else
    fprintf('Skip Table 28: %s missing (run separability_all).\n', sf);
end

%% ---- Table 5: relative NMI change between smallest and largest label ratio ----
lf = fullfile(resultsDir, 'sens_label_all.mat');
if exist(lf, 'file')
    L = load(lf);                       % allNMI: dataset x ratio x seed
    fid = fopen(fullfile(outDir, 'table5_rel_change_label_ratio.csv'), 'w');
    fprintf(fid, 'Dataset,NMI_at_%d%%,NMI_at_%d%%,RelChange%%\n', round(100*L.ratios(1)), round(100*L.ratios(end)));
    for di = 1:numel(L.datasets)
        mu = squeeze(mean(L.allNMI(di, :, :), 3, 'omitnan'));
        if all(isnan(mu)), continue; end
        fprintf(fid, '%s,%.4f,%.4f,%.2f\n', L.datasets{di}, mu(1), mu(end), ...
            100 * (mu(end) - mu(1)) / max(mu(1), eps));
    end
    fclose(fid);
    fprintf('Wrote Table 5.\n');
else
    fprintf('Skip Table 5: %s missing (run sensitivity_label_all).\n', lf);
end
fprintf('Done. Tables in %s\n', outDir);
