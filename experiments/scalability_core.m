function scalability_core(datasets, outName)
% SCALABILITY_CORE  Measured scalability of PSAE-SSL (train/infer time, peak memory)
%   over usr.numSeeds seeds for the given dataset list. Each run = fresh MATLAB process.
%   Called by scalability_all.
%   Number of seeds, label fraction, epochs etc. come from src/utils/get_user_settings.m.
%   Per-seed cache files live in the results folder that belongs to the active
%   settings (get_results_dir.m), so a changed epoch/lr/... setting never reuses
%   old measurements, while raising the number of seeds only adds the new seeds.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));
expDir     = fullfile(projectRoot, 'experiments');
resultsDir = get_results_dir(projectRoot);
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end
usr       = get_user_settings();
numRuns   = usr.numSeeds;
labelFrac = usr.labelFrac;
ensure_datasets(projectRoot, datasets, usr);   % downloads from Zenodo if missing

% One fresh MATLAB process per run (matches the protocol described in the
% paper). Falls back to in-process runs if the MATLAB binary is not found.
useFreshProcess = true;
matlabBin = fullfile(matlabroot, 'bin', 'matlab');
if ispc, matlabBin = [matlabBin '.exe']; end
if useFreshProcess && ~exist(matlabBin, 'file')
    warning('MATLAB binary not found at %s; running in-process (peak memory will include earlier runs).', matlabBin);
    useFreshProcess = false;
end

% Record the hardware/software context so it can be reported with the table.
env = struct('computer', computer, 'release', version('-release'), ...
    'threads', maxNumCompThreads, 'freshProcess', useFreshProcess);
fprintf('Environment: %s | MATLAB %s | %d threads | freshProcess=%d\n', ...
    env.computer, env.release, env.threads, env.freshProcess);

summary = struct();
for di = 1:numel(datasets)
    name = datasets{di};
    folder = fullfile(projectRoot, 'data', name);
    if ~exist(folder, 'dir')
        warning('[%s] data folder missing - skip', name); continue
    end
    cfg = get_dataset_config(name);
    try
        [adjList, allLabels, K] = load_dataset(name, fullfile(projectRoot, 'data'));
    catch ME
        warning('[%s] load failed: %s - skip', name, ME.message); continue
    end
    n = numel(adjList);
    fprintf('\n=== Scalability PSAE-SSL | %s | n=%d K=%d | %d runs ===\n', name, n, K, numRuns);

    trainT  = nan(numRuns, 1);
    inferT  = nan(numRuns, 1);
    peakMB  = nan(numRuns, 1);
    baseMB  = nan(numRuns, 1);
    deltaMB = nan(numRuns, 1);
    arrayMB = nan(numRuns, 1);

    for s = 1:numRuns
        outFile = fullfile(resultsDir, sprintf('scal_psaessl_%s_s%d.mat', name, s));

        % Reuse a cached run only if it holds measured values. Files written
        % by older versions of this script lack the 'measured' flag and are
        % ignored (they contained reference-table values, not measurements).
        if exist(outFile, 'file')
            C = load(outFile, 'R'); C = C.R;
            if isfield(C, 'ok') && C.ok && isfield(C, 'measured') && C.measured
                R = C;
                trainT(s) = R.train; inferT(s) = R.infer; peakMB(s) = R.peakMB;
                baseMB(s) = R.baseMB; deltaMB(s) = R.deltaMB;
                arrayMB(s) = R.arrayMB;
                fprintf('  seed=%2d  (cached) train=%.2fs infer=%.3fs peak=%.0fMB\n', ...
                    s, R.train, R.infer, R.peakMB);
                continue
            end
        end

        R = struct('method', 'PSAE-SSL', 'dataset', name, 'seed', s, 'ok', false, ...
            'measured', false, 'train', NaN, 'infer', NaN, 'peakMB', NaN, ...
            'baseMB', NaN, 'deltaMB', NaN, 'arrayMB', NaN, ...
            'err', '', 'memKind', '');
        try
            if useFreshProcess
                if exist(outFile, 'file'); delete(outFile); end
                cmd = sprintf('"%s" -batch "addpath(''%s''); scal_worker(''PSAE-SSL'',''%s'',%d,''%s'')"', ...
                    matlabBin, expDir, name, s, outFile);
                % '-echo' streams the worker's [Stage1]/[Stage2]/[Stage3] log live.
                [st, out] = system(cmd, '-echo');
                if exist(outFile, 'file')
                    W = load(outFile, 'R'); R = W.R;
                else
                    error('worker produced no result (exit status %d); see the log above', st);
                end
                if ~(isfield(R, 'ok') && R.ok)
                    error('worker failed: %s', R.err);
                end
            else
                [R.baseMB, R.memKind] = get_peak_mem_mb();
                r = psae_pipeline(adjList, allLabels, K, cfg, s, ...
                    struct('labelFrac', labelFrac, 'stratified', false, 'keepEmb', true));
                R.train = r.trainTime;
                R.infer = r.inferTime;
                if isfield(r, 'Z0') && isfield(r, 'Hf')
                    Z0 = r.Z0; Hf = r.Hf; %#ok<NASGU>
                    info = whos('Z0', 'Hf');
                    R.arrayMB = sum([info.bytes]) / 1048576;
                else
                    R.arrayMB = 0;
                end
                [R.peakMB, ~] = get_peak_mem_mb();
                R.deltaMB = max(0, R.peakMB - R.baseMB);
                R.ok = true;
            end
            R.measured = true;
            fprintf('  seed=%2d  train=%.2fs  infer=%.3fs  peak=%.0fMB\n', ...
                s, R.train, R.infer, R.peakMB);
        catch ME
            R.err = ME.message;
            warning('  seed=%d FAILED: %s', s, ME.message);
        end
        save(outFile, 'R');

        trainT(s) = R.train; inferT(s) = R.infer; peakMB(s) = R.peakMB;
        baseMB(s) = R.baseMB; deltaMB(s) = R.deltaMB;
        arrayMB(s) = R.arrayMB;
    end

    summary.(name) = struct( ...
        'n', n, 'K', K, ...
        'train_mean', mean(trainT, 'omitnan'), 'train_std', std(trainT, 0, 'omitnan'), ...
        'infer_mean', mean(inferT, 'omitnan'), 'infer_std', std(inferT, 0, 'omitnan'), ...
        'peak_mean',  mean(peakMB, 'omitnan'), 'peak_std',  std(peakMB, 0, 'omitnan'), ...
        'delta_mean', mean(deltaMB, 'omitnan'), 'delta_std', std(deltaMB, 0, 'omitnan'), ...
        'array_mean', mean(arrayMB, 'omitnan'), 'array_std', std(arrayMB, 0, 'omitnan'), ...
        'trainT', trainT, 'inferT', inferT, 'peakMB', peakMB);
end

% keep earlier summaries of datasets that are not part of this run
outMat = fullfile(resultsDir, [outName '.mat']);
if exist(outMat, 'file')
    try
        Old = load(outMat, 'summary');
        ofn = fieldnames(Old.summary);
        for i = 1:numel(ofn)
            if ~isfield(summary, ofn{i}), summary.(ofn{i}) = Old.summary.(ofn{i}); end
        end
    catch
    end
end
datasets = fieldnames(summary)'; %#ok<NASGU>
save(outMat, 'summary', 'datasets', 'numRuns', 'env');

fprintf('\n===== Measured summary (mean +/- std over %d seeds) =====\n', numRuns);
fprintf('%-10s %20s %20s %20s\n', 'Dataset', 'Train (s)', 'Infer (s)', 'Peak (MB)');
fn = fieldnames(summary);
for i = 1:numel(fn)
    q = summary.(fn{i});
    fprintf('%-10s %9.2f +/- %-8.2f %9.3f +/- %-8.3f %9.0f +/- %-6.0f\n', fn{i}, ...
        q.train_mean, q.train_std, q.infer_mean, q.infer_std, q.peak_mean, q.peak_std);
end
fprintf('\nSaved results/%s.mat\n', outName);
end
