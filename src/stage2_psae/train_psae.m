function [model, history] = train_psae(Z0, adjList, labeledIdx, labeledLabels, K, hp)
% TRAIN_PSAE  Stage 2 of PSAE-SSL, end to end: build the parallel
%             stacked autoencoder (PSAE_INIT), pretrain it in an
%             unsupervised manner (Algorithm 3, first loop), then
%             fine-tune it semi-supervised (Algorithm 3, second loop)
%             with periodic pseudo-label refresh (the dedicated
%             "pseudo-label generation and refresh" algorithm).
%
%   [model, history] = TRAIN_PSAE(Z0, adjList, labeledIdx, labeledLabels, K, hp)
%
%   Inputs
%     Z0            : n x d Stage-1 (Node2Vec) embedding matrix Z^(0)
%     adjList       : n x 1 adjacency-list cell array (needed for the
%                     AHC calls inside pseudo-label generation)
%     labeledIdx    : 1 x |L| ground-truth labeled node indices
%     labeledLabels : 1 x |L| corresponding class labels in {1..K}
%     K             : number of community classes
%     hp            : hyperparameter struct, all fields optional
%                     (defaults reproduce Section "Implementation
%                     details": lr=1e-3, weightDecay=1e-4, up to 1000
%                     epochs, patience=10, valFrac=0.15, pseudo-label
%                     T=5, retention 20%; architectural fields such as
%                     branchLatentSizes are dataset-dependent per
%                     Table "Overview of benchmark datasets" and are
%                     set by the caller per dataset)
%
%   Outputs
%     model   : trained PSAE model struct (see PSAE_INIT / PSAE_FORWARD)
%     history : struct with pretrainLoss, finetuneLoss, finetuneValLoss
%               (for diagnostic plots) and the final fused embeddings Hf
%               (dF x n)

if nargin < 6
    hp = struct();
end
hp = set_default(hp, 'branchLatentSizes', {[64, 32], [64, 32], [64, 32]}); % B=3 branches
hp = set_default(hp, 'fusionDim', 32);
hp = set_default(hp, 'outputActivation', 'linear');
hp = set_default(hp, 'lambdaRec', 1);     % Table "Key parameters": lambda_rec = 1
hp = set_default(hp, 'lambdaSup', 0.5);   % lambda_sup = 0.5
hp = set_default(hp, 'lambdaSep', 1);     % lambda_sep = 1
hp = set_default(hp, 'margin', 1.0);
hp = set_default(hp, 'maxEpochsPretrain', 1000); % Paper: up to 1000 epochs (P3)
hp = set_default(hp, 'maxEpochsFinetune', 1000); % Section 4.4: up to 1000 epochs
hp = set_default(hp, 'patience', 10);            % Section 4.4: early stopping patience
hp = set_default(hp, 'lr', 1e-3);                % Section 4.4: Adam lr = 1e-3
hp = set_default(hp, 'beta1', 0.9);
hp = set_default(hp, 'beta2', 0.999);
hp = set_default(hp, 'epsAdam', 1e-8);
hp = set_default(hp, 'weightDecay', 1e-4);       % Section 4.4: weight decay = 1e-4
hp = set_default(hp, 'valFrac', 0.15);           % Section 4.4: ~15% of labeled nodes
hp = set_default(hp, 'pseudoT', 5);              % Algorithm: refresh interval T = 5 epochs
hp = set_default(hp, 'pseudoRetain', 0.20);      % Algorithm: per-cluster retention p = 20%
hp = set_default(hp, 'seed', 1);

rng(hp.seed);

X = Z0';                     % dInput x n (Stage-2 convention: columns = nodes)
[dInput, n] = size(X);

if isfield(hp,'batchSize') && ~isempty(hp.batchSize) && hp.batchSize > 0 && n > hp.batchSize
    [model, history] = train_psae_minibatch(X, adjList, labeledIdx, labeledLabels, K, hp);
    return
end

model = psae_init(dInput, hp.branchLatentSizes, hp.fusionDim, K, hp.outputActivation);
% Specialize branch masks with labeled nodes (class-specific design)
model = apply_class_masks(model, X, labeledIdx, labeledLabels);
moments = init_adam_moments(model);

history.pretrainLoss = [];
history.finetuneLoss = [];
history.finetuneValLoss = [];

t = 0;
emptySep = cell(1, K);

%% ---- Phase 1: unsupervised pretraining (Algorithm 3, first loop) ----
fprintf('[Stage2:pretrain] start (maxEpochs=%d)\n', hp.maxEpochsPretrain);
lambda0 = struct('rec', hp.lambdaRec, 'sup', 0, 'sep', 0);
bestLoss = Inf;
noImprove = 0;
for epoch = 1:hp.maxEpochsPretrain
    t = t + 1;
    out = psae_forward(model, X);
    losses = compute_losses(out, X, [], [], emptySep, lambda0, hp.margin);
    grads  = psae_backward(model, out, X, [], [], emptySep, lambda0, hp.margin);
    [model, moments] = apply_adam_to_model(model, grads, moments, t, adam_hp(hp));

    history.pretrainLoss(end + 1) = losses.rec;

    if losses.rec < bestLoss - 1e-6
        bestLoss = losses.rec;
        noImprove = 0;
    else
        noImprove = noImprove + 1;
    end
    if noImprove >= hp.patience
        fprintf('[Stage2:pretrain] early stop at epoch %d (rec=%.5f)\n', epoch, losses.rec);
        break
    end
end
fprintf('[Stage2:pretrain] done (epochs=%d, rec=%.5f)\n', numel(history.pretrainLoss), bestLoss);

%% ---- Stratified train/validation split of the ground-truth labeled set ----
[trainIdx, valIdx] = stratified_split(labeledIdx, labeledLabels, hp.valFrac, hp.seed);
trainLabelsFull = zeros(1, n);
trainLabelsFull(labeledIdx) = labeledLabels;

%% ---- Initial pseudo-labels (right after pretraining) ----
out = psae_forward(model, X);
[P_idx, P_label] = generate_pseudo_labels(out.Hf, adjList, labeledIdx, labeledLabels, hp.pseudoRetain, K);
classIdxCellForSep = build_class_idx_cell(trainIdx, trainLabelsFull(trainIdx), P_idx, P_label, K);

%% ---- Phase 2: semi-supervised fine-tuning with pseudo-label refresh ----
fprintf('[Stage2:finetune] start (maxEpochs=%d)\n', hp.maxEpochsFinetune);
lambda = struct('rec', hp.lambdaRec, 'sup', hp.lambdaSup, 'sep', hp.lambdaSep);
noImprove = 0;
bestModel = model;
% [v7 fix] The pretrained model is itself a candidate (evaluated before any
% fine-tune step), so a collapsed / worsened fine-tune can never replace it.
outP = psae_forward(model, X);
vl0 = compute_losses(outP, X, trainLabelsFull, valIdx, classIdxCellForSep, lambda, hp.margin);
ahpFt = adam_hp(hp);
warmEp = 20;
if isfield(hp, 'sepWarmupEpochs') && ~isempty(hp.sepWarmupEpochs), warmEp = hp.sepWarmupEpochs; end
bestVal = vl0.total;
% [v7] Gentler fine-tuning: lower lr, fresh Adam state, sep-loss warm-up.
if isfield(hp, 'ftLrScale') && ~isempty(hp.ftLrScale), ahpFt.lr = ahpFt.lr * hp.ftLrScale; else, ahpFt.lr = ahpFt.lr * 0.1; end
moments = init_adam_moments(model);
t = 0;

for epoch = 1:hp.maxEpochsFinetune
    t = t + 1;
    lamE = lambda; lamE.sep = lambda.sep * min(1, epoch / max(warmEp, 1));

    if epoch > 1 && mod(epoch, hp.pseudoT) == 0
        out = psae_forward(model, X);
        [P_idx, P_label] = generate_pseudo_labels(out.Hf, adjList, labeledIdx, labeledLabels, hp.pseudoRetain, K);
        classIdxCellForSep = build_class_idx_cell(trainIdx, trainLabelsFull(trainIdx), P_idx, P_label, K);
    end

    out = psae_forward(model, X);
    trainLosses = compute_losses(out, X, trainLabelsFull, trainIdx, classIdxCellForSep, lamE, hp.margin);
    grads = psae_backward(model, out, X, trainLabelsFull, trainIdx, classIdxCellForSep, lamE, hp.margin);
    [model, moments] = apply_adam_to_model(model, grads, moments, t, ahpFt);

    out = psae_forward(model, X); % recompute with updated params for the validation read-out
    valLosses = compute_losses(out, X, trainLabelsFull, valIdx, classIdxCellForSep, lambda, hp.margin);

    history.finetuneLoss(end + 1) = trainLosses.total;
    history.finetuneValLoss(end + 1) = valLosses.total;

    if valLosses.total < bestVal - 1e-6
        bestVal = valLosses.total;
        noImprove = 0;
        bestModel = model;
    else
        noImprove = noImprove + 1;
    end
    minFt = 20;
    if isfield(hp, 'minEpochsFinetune') && ~isempty(hp.minEpochsFinetune)
        minFt = hp.minEpochsFinetune;
    end
    if noImprove >= hp.patience && epoch >= minFt
        fprintf('[Stage2:finetune] early stop at epoch %d (val total=%.5f)\n', epoch, valLosses.total);
        break
    end
end

model = bestModel; % restore the best-on-validation snapshot
fprintf('[Stage2:finetune] done (epochs=%d, val=%.5f)\n', numel(history.finetuneLoss), bestVal);
out = psae_forward(model, X);
history.Hf = out.Hf; % dF x n, final fused embeddings H^(f)
end

function hp = set_default(hp, field, val)
if ~isfield(hp, field) || isempty(hp.(field))
    hp.(field) = val;
end
end

function s = adam_hp(hp)
s = struct('lr', hp.lr, 'beta1', hp.beta1, 'beta2', hp.beta2, ...
    'eps', hp.epsAdam, 'weightDecay', hp.weightDecay);
end
