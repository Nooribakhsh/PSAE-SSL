function scal_worker(method, dataset, seed, outFile)
% SCAL_WORKER  One scalability run (called by scalability_core in a fresh MATLAB process).
%
%   scal_worker('PSAE-SSL', 'karate', 1, 'results/scal_psaessl_karate_s1.mat')
%   All settings (epochs, label fraction, ...) are read from src/utils/get_user_settings.m.

R = struct('method', method, 'dataset', dataset, 'seed', seed, 'ok', false, ...
    'train', NaN, 'infer', NaN, 'peakMB', NaN, 'baseMB', NaN, 'deltaMB', NaN, ...
    'arrayMB', NaN, 'err', '', 'memKind', '', 'measured', false);
try
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(projectRoot, 'src')));
    if ~strcmpi(method, 'PSAE-SSL')
        error('Only method PSAE-SSL is supported.');
    end
    usr = get_user_settings();
    cfg = get_dataset_config(dataset);
    [adjList, allLabels, K] = load_dataset(dataset, fullfile(projectRoot, 'data'));
    [R.baseMB, R.memKind] = get_peak_mem_mb();
    r = psae_pipeline(adjList, allLabels, K, cfg, seed, ...
        struct('labelFrac', usr.labelFrac, 'stratified', false, 'keepEmb', true));
    R.train = r.trainTime; R.infer = r.inferTime;
    if isfield(r, 'Z0') && isfield(r, 'Hf')
        Z0 = r.Z0; Hf = r.Hf; %#ok<NASGU>
        info = whos('Z0', 'Hf'); R.arrayMB = sum([info.bytes]) / 1048576;
    end
    [R.peakMB, ~] = get_peak_mem_mb();
    R.deltaMB = R.peakMB - R.baseMB;
    R.ok = true;
    R.measured = true;
catch ME
    R.err = ME.message;
end
save(outFile, 'R');
end
