function r = psae_pipeline(adjList, allLabels, K, cfg, seed, opts)
% PSAE_PIPELINE  One full PSAE-SSL run (Stage 1 -> 2 -> 3 -> metrics).
% Protocol locked to paper Section 4.4 and Tables 2-3:
%   - 10% labels via UNIFORM random sampling (not stratified) by default
%   - Adam lr=1e-3, weightDecay=1e-4, patience=10, valFrac=0.15
%   - lambda_rec=1, lambda_sup=0.5, lambda_sep=1, margin=1, B=3
%   - same labeled split seed used for all methods in a given run
%
%   r = PSAE_PIPELINE(adjList, allLabels, K, cfg, seed, opts)
%
%   opts (all optional):
%     .variant      'full' (default), '-Node2Vec', 'SAE', '-sep', '-SSL',
%                   'k-means'
%     .B            number of branches (default cfg.B or 3)
%     .labelFrac    labeled fraction (default 0.10)
%     .stratified   true ONLY for sensitivity Table 30 / stratified analysis;
%                   main tables use false (uniform sampling)
%     .Z0           precomputed Node2Vec embedding (n x d), skips Stage 1
%     .computeSep   true -> also compute FDR/SC/DBI/CHI before/after
%     .keepEmb      true -> return Z0 and Hf
%     .labelGuided  true -> after Stage 3, refine the partition with the labeled
%                   nodes (prototype + graph/embedding label propagation, see
%                   label_guided_clustering.m). Default false (paper AHC only).
%     .psaeMode     'v7' (default) | 'v6' (old fine-tune) | 'none' (skip Stage 2)
%     .forceTargetK false (default, paper-aligned) -> AHC + modularity-optimal
%                   dendrogram cut (Section 3.3 / Algorithm 5). Set true only for
%                   sensitivity tables that require a fixed ground-truth K.

if nargin < 6, opts = struct(); end
variant    = getf(opts, 'variant', 'full');
B          = getf(opts, 'B', getf(cfg, 'B', 3));
labelFrac  = getf(opts, 'labelFrac', 0.10);
% CRITICAL: paper main experiments use uniform (non-stratified) sampling
stratified = getf(opts, 'stratified', false);
Z0given    = getf(opts, 'Z0', []);
computeSep = getf(opts, 'computeSep', false);
keepEmb    = getf(opts, 'keepEmb', false);
labelGuided = getf(opts, 'labelGuided', false);
% Paper Section 3.3: Stage 3 selects the number of communities by cutting the
% AHC dendrogram at the modularity-maximizing level (forceTargetK=false).
% Set forceTargetK=true only for sensitivity / fixed-K evaluation experiments.
forceTargetK = getf(opts, 'forceTargetK', false);

n = numel(adjList);
usr = get_user_settings();   % settings: get_user_settings.m

% ---- labeled set (Section 4.4: uniform random over full node set) ----
if stratified
    [labeledIdx, labeledLabels] = sample_labeled_set_stratified(allLabels, labelFrac, seed);
else
    [labeledIdx, labeledLabels] = sample_labeled_set(allLabels, labelFrac, seed);
end

% ---- Stage 1 ----
tTrain = tic;
sgUsed = NaN;   % Skip-gram epochs actually used (logged below)
switch variant
    case '-Node2Vec'
        % Sparse adjacency + fixed-dim projection (E4). targetDim matches
        % Node2Vec d so layer widths in Table 2 stay valid (N = d).
        dN2V = 64;
        if isfield(cfg, 'n2v') && isfield(cfg.n2v, 'd')
            dN2V = cfg.n2v.d;
        end
        Z0 = adjlist_to_features(adjList, dN2V, seed);
    otherwise
        if isempty(Z0given)
            n2v = cfg.n2v;
            bud0 = get_training_budget(getf(cfg, 'name', ''), n);
            n2v.numEpochs = bud0.skipgramEpochs;   % epoch budget (usr.epochs)
            sgUsed = n2v.numEpochs;
            n2v.seed = seed;
            Z0 = node2vec(adjList, n2v);
        else
            Z0 = Z0given;
        end
end

% ---- Stage 2 ----
lam = getf(cfg, 'lambda', struct('rec',1,'sup',0.5,'sep',0.25));
hp = struct();
hp.branchLatentSizes = repmat({cfg.layers}, 1, B);
hp.fusionDim         = cfg.fusionDim;
hp.outputActivation  = 'linear';
hp.lambdaRec         = getf(lam, 'rec', 1);
hp.lambdaSup         = getf(lam, 'sup', 0.5);
hp.lambdaSep         = getf(lam, 'sep', 1);
hp.margin            = getf(cfg, 'margin', 1.0);
hp.patience          = usr.patience;
hp.lr                = usr.lr;
hp.weightDecay       = getf(cfg, 'wd', 1e-4);
hp.valFrac           = usr.valFrac;
hp.pseudoT           = usr.pseudoT;
hp.pseudoRetain      = usr.pseudoRetain;   % share closest to the centroid
hp.ftLrScale         = usr.ftLrScale;
hp.sepWarmupEpochs   = usr.sepWarmupEpochs;
hp.seed              = seed;
hp.B                 = B;
% Adam hyperparameters (required by apply_adam_to_model / adam_hp)
hp.beta1             = usr.beta1;
hp.beta2             = usr.beta2;
hp.epsAdam           = usr.epsAdam;
% Epoch budgets: usr.epochs in get_user_settings.m
bud = get_training_budget(getf(cfg, 'name', ''), n);
hp.maxEpochsPretrain = bud.maxEpochsPretrain;
hp.maxEpochsFinetune = bud.maxEpochsFinetune;
hp.minEpochsFinetune = bud.minEpochsFinetune;
if isfield(cfg, 'batchSize'), hp.batchSize = cfg.batchSize; end

switch variant
    case 'SAE'
        hp.branchLatentSizes = {cfg.layers};
    case '-sep'
        hp.lambdaSep = 0;
    case '-SSL'
        % Paper: "remove the semi-supervised objective (λ_sup = 0)".
        % Keep fine-tuning (L_rec + L_sep + pseudo-labels) active (E3).
        hp.lambdaSup = 0;
end

% Always call train_psae so Adam defaults (beta1/beta2/epsAdam) are applied.
% train_psae dispatches to train_psae_minibatch when batchSize > 0 and n is large.
fprintf('[Stage2] PSAE parallel stacked AE (B=%d) | sgEpochs=%d, capPre=%d, capFt=%d\n', ...
    B, sgUsed, hp.maxEpochsPretrain, hp.maxEpochsFinetune);
% opts.psaeMode (ablation): 'v7' (default) | 'v6' (legacy fine-tune) | 'none'
%   'none' skips Stage 2 entirely: Stage 3 / label stage run on Node2Vec Z0.
psaeMode = getf(opts, 'psaeMode', 'v7');
switch psaeMode
    case 'none'
        fprintf('[Stage2] SKIPPED (psaeMode=none): using Node2Vec embedding directly\n');
        psaeModel = []; stage2History = struct('Hf', Z0');
    case 'v6'
        hp.legacyFinetune = true;
        [psaeModel, stage2History] = train_psae(Z0, adjList, labeledIdx, labeledLabels, K, hp);
    otherwise
        [psaeModel, stage2History] = train_psae(Z0, adjList, labeledIdx, labeledLabels, K, hp);
end
Hf = stage2History.Hf;
trainTime = toc(tTrain);

% ---- Stage 3: Agglomerative Hierarchical Clustering (AHC) ----
% Paper Algorithm 4/5: build dendrogram on fused embeddings H^(f),
% cut at modularity-maximizing level, then optional post-cut
% inverse-distance reassignment to refine community boundaries.
dimF = size(Hf, 1);
if size(Hf, 2) == n, dimF = size(Hf, 1); elseif size(Hf, 1) == n, dimF = size(Hf, 2); end
fprintf('[Stage3] AHC on fused embeddings H^(f) (%d nodes, dim=%d)...\n', n, dimF);
tInfer = tic;
if size(Hf, 2) ~= n && size(Hf, 1) == n
    Hf = Hf';
end
colN = sqrt(sum(Hf.^2, 1)) + 1e-12;
Hf = Hf ./ colN;

if strcmp(variant, 'k-means')
    fprintf('[Stage3] Variant=k-means  ->  clustering with K=%d\n', K);
    predLabels = simple_kmeans(Hf, K, seed);
    bestK = K;
elseif forceTargetK && ~isempty(K) && K >= 2
    % Sensitivity / fixed-K path only: AHC dendrogram cut at ground-truth K.
    % No k-means fallback — paper Stage 3 is AHC throughout (Section 3.3).
    fprintf('[Stage3] forceTargetK=true (n=%d): AHC cut targeting K=%d\n', n, K);
    predLabels = cluster_embeddings(Hf, K, seed, adjList);
    if numel(unique(predLabels)) ~= K
        fprintf('[Stage3]   retry with seed+11 (still AHC, no k-means)...\n');
        predLabels = cluster_embeddings(Hf, K, seed + 11, adjList);
    end
    if numel(unique(predLabels)) ~= K
        warning('psae_pipeline:ahcK', ...
            ['AHC could not reach exactly K=%d (got %d communities). ' ...
             'Keeping the AHC partition; k-means fallback is disabled for paper alignment.'], ...
            K, numel(unique(predLabels)));
    end
    bestK = numel(unique(predLabels));
    predLabels = predLabels(:);
else
    % Paper default for all graph sizes: adjacency-constrained single-linkage
    % AHC + modularity-optimal dendrogram cut + post-cut reassignment
    fprintf('[Stage3] AHC dendrogram + modularity cut (n=%d, paper-aligned)\n', n);
    [~, repMerges, ~] = ahc(Hf, adjList);
    minK = max(1, n - size(repMerges, 1));
    Krange = minK:min(n, max(minK + 200, 300));
    fprintf('[Stage3]   selecting cut over K in [%d, %d] by max modularity Q\n', ...
        Krange(1), Krange(end));
    [predLabels, bestK] = select_k_by_modularity(repMerges, n, adjList, Krange);
    fprintf('[Stage3]   modularity-optimal K=%d; reassigning isolates...\n', bestK);
    predLabels = reassign_isolates(Hf, predLabels, bestK);
    fprintf('[Stage3]   post-cut inverse-distance reassignment (Algorithm 6)...\n');
    predLabels = post_cut_reassignment(Hf, adjList, predLabels, 1e-6);
end
% [v7] label-guided scores are dense n x K matrices: skip them when that would
% not fit in memory (e.g. DBLP with K=13477); the unsupervised partition is kept.
lgTooBig = (double(n) * double(max(K, 1)) > 6e7);
if labelGuided && lgTooBig
    fprintf('[Stage3] label-guided refinement skipped (n*K=%.1e too large for dense scores)\n', double(n)*double(K));
end
if labelGuided && ~lgTooBig && ~isempty(labeledIdx) && ~strcmp(variant, 'k-means')
    fprintf('[Stage3] label-guided refinement (%d labeled nodes)...\n', numel(labeledIdx));
    classList = unique(allLabels(:))';
    % [v7] pick Hf / [Hf;Z0] / Z0 by hold-out accuracy on LABELED nodes only
    if strcmp(psaeMode, 'v7')
        [HfLG, selInfo] = select_label_embedding(Hf, Z0, adjList, labeledIdx, labeledLabels, ...
            classList, predLabels, seed);
    else   % 'v6' / 'none': use the embedding as is (no selection)
        HfLG = Hf;  selInfo = struct('choice', 1, 'acc', nan(1,3), 'collapsed', false);
    end
    if selInfo.collapsed || selInfo.choice ~= 1
        fprintf('[Stage3]   embedding for label stage: candidate %d (collapsed Hf=%d, holdout acc=%s)\n', ...
            selInfo.choice, selInfo.collapsed, mat2str(round(selInfo.acc, 3)));
    end
    predLabels = label_guided_clustering(HfLG, adjList, labeledIdx, labeledLabels, ...
        classList, predLabels, seed);
    predLabels = predLabels(:);
end
inferTime = toc(tInfer);
nComm = numel(unique(predLabels));
fprintf('[Stage3] Done. Predicted communities=%d (true K=%d), inferTime=%.2fs\n', ...
    nComm, K, inferTime);

% ---- Metrics ----
dsName = '';
if isfield(cfg, 'name'), dsName = cfg.name; end
if isfield(cfg, 'dataset'), dsName = cfg.dataset; end
m = evaluate_partition(adjList, allLabels, predLabels, dsName);
r = m;
r.bestK      = bestK;       % modularity-selected #communities (or K for k-means)
r.trueK      = K;           % ground-truth K (for reference only; not used in cut)
r.predLabels = predLabels;
r.labeledIdx = labeledIdx;
r.trainTime  = trainTime;
r.inferTime  = inferTime;
r.seed       = seed;
r.variant    = variant;
r.B          = B;
r.labelFrac  = labelFrac;
r.stratified = stratified;
r.labelGuided = labelGuided;
r.psaeMode = psaeMode;

if computeSep
    % Paper §5.1.3: use ground-truth community labels for intra/inter-class
    % statistics, independent of the clustering algorithm (E2).
    r.sepAfter  = separability_metrics(Hf, allLabels);
    Z0t = Z0';
    r.sepBefore = separability_metrics(Z0t, allLabels);
end
if keepEmb
    r.Z0 = Z0;
    r.Hf = Hf;
end
end

function v = getf(s, f, d)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f))
    v = s.(f);
else
    v = d;
end
end
