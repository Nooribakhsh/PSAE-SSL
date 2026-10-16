function [model, history] = train_psae_minibatch(X, adjList, labeledIdx, labeledLabels, K, hp)
% TRAIN_PSAE_MINIBATCH  Mini-batch version of TRAIN_PSAE (same algorithm,
%   same losses, same pseudo-label refresh), for PubMed / DBLP-sized graphs.
%
%   X  : dInput x n   (columns = nodes), hp : struct already filled with
%   defaults by TRAIN_PSAE (this file is called from there).
%   Extra hp fields (optional):
%     batchSize          (required, e.g. 128 or 256)
%     maxBatchesPerEpoch (default Inf) -> cap steps per epoch on huge graphs
%     labeledPerBatch    (default floor(batchSize/4)) -> train-labeled nodes
%                        appended to every batch so L_sup / L_sep are defined
%     evalChunk          (default 4096) -> chunk size for full-graph evaluation

bs     = hp.batchSize;
tStage2 = tic;
maxBpe = getf(hp, 'maxBatchesPerEpoch', Inf);
nLab   = getf(hp, 'labeledPerBatch', floor(bs / 4));
chunk  = getf(hp, 'evalChunk', 4096);

% Ensure Adam fields exist even if caller skipped train_psae defaults
hp.lr          = getf(hp, 'lr', 1e-3);
hp.beta1       = getf(hp, 'beta1', 0.9);
hp.beta2       = getf(hp, 'beta2', 0.999);
hp.epsAdam     = getf(hp, 'epsAdam', 1e-8);
hp.weightDecay = getf(hp, 'weightDecay', 1e-4);
hp.margin      = getf(hp, 'margin', 1.0);
hp.valFrac     = getf(hp, 'valFrac', 0.15);
hp.patience    = getf(hp, 'patience', 10);
hp.pseudoT     = getf(hp, 'pseudoT', 5);
hp.pseudoRetain = getf(hp, 'pseudoRetain', 0.20);
hp.seed        = getf(hp, 'seed', 1);
hp.maxEpochsPretrain = getf(hp, 'maxEpochsPretrain', 1000);
hp.maxEpochsFinetune = getf(hp, 'maxEpochsFinetune', 1000);
hp.lambdaRec   = getf(hp, 'lambdaRec', 1);
hp.lambdaSup   = getf(hp, 'lambdaSup', 0.5);
hp.lambdaSep   = getf(hp, 'lambdaSep', 1);

[~, n] = size(X);
dInput = size(X, 1);

model   = psae_init(dInput, hp.branchLatentSizes, hp.fusionDim, K, hp.outputActivation);
model   = apply_class_masks(model, X, labeledIdx, labeledLabels);
moments = init_adam_moments(model);
ahp     = adam_hp(hp);

history.pretrainLoss    = [];
history.finetuneLoss    = [];
history.finetuneValLoss = [];
t = 0;

%% ---- split labeled set ----
[trainIdx, valIdx] = stratified_split(labeledIdx, labeledLabels, hp.valFrac, hp.seed);
trainLabelsFull = zeros(1, n);
trainLabelsFull(labeledIdx) = labeledLabels;
isTrain = false(1, n);
isTrain(trainIdx) = true;
trainLabOnly = trainLabelsFull .* isTrain;   % labels visible to L_sup

%% ---- Phase 1: unsupervised pretraining ----
lambda0 = struct('rec', hp.lambdaRec, 'sup', 0, 'sep', 0);
emptySep = cell(1, K);
bestLoss = Inf; noImprove = 0;
fprintf('[Stage2:pretrain] start (maxEpochs=%d)\n', hp.maxEpochsPretrain);
for epoch = 1:hp.maxEpochsPretrain
    perm = randperm(n);
    nB = min(ceil(n / bs), maxBpe);
    epochLoss = 0;
    for bi = 1:nB
        idx = perm((bi - 1) * bs + 1 : min(bi * bs, n));
        Xb = X(:, idx);
        t = t + 1;
        out = psae_forward(model, Xb);
        l = compute_losses(out, Xb, [], [], emptySep, lambda0, hp.margin);
        g = psae_backward(model, out, Xb, [], [], emptySep, lambda0, hp.margin);
        [model, moments] = apply_adam_to_model(model, g, moments, t, ahp);
        epochLoss = epochLoss + l.rec;
    end
    epochLoss = epochLoss / nB;
    history.pretrainLoss(end + 1) = epochLoss;
    if mod(epoch, 25) == 0
        fprintf('[Stage2:pretrain] epoch %d/%d rec=%.5f (%.0fs)\n', epoch, hp.maxEpochsPretrain, epochLoss, toc(tStage2));
    end
    if epochLoss < bestLoss - 1e-6
        bestLoss = epochLoss; noImprove = 0;
    else
        noImprove = noImprove + 1;
    end
    if noImprove >= hp.patience
        fprintf('[Stage2-mb:pretrain] early stop at epoch %d (rec=%.5f)\n', epoch, epochLoss);
        break
    end
end

fprintf('[Stage2:pretrain] done (epochs=%d, rec=%.5f)\n', numel(history.pretrainLoss), history.pretrainLoss(end));

%% ---- initial pseudo-labels ----
Hf = forward_hf(model, X, chunk);
[P_idx, P_label] = generate_pseudo_labels(Hf, adjList, labeledIdx, labeledLabels, hp.pseudoRetain, K);
classIdxGlobal = build_class_idx_cell(trainIdx, trainLabelsFull(trainIdx), P_idx, P_label, K);

%% ---- Phase 2: semi-supervised fine-tuning ----
lambda = struct('rec', hp.lambdaRec, 'sup', hp.lambdaSup, 'sep', hp.lambdaSep);
noImprove = 0; bestModel = model;
map = zeros(1, n);
% [v7 fix] The pretrained model is itself a candidate: evaluate it BEFORE any
% fine-tune step, so a collapsed / worsened fine-tune can never replace it.
% hp.legacyFinetune = true reproduces the v6 behaviour exactly (for ablation).
legacy = getf(hp, 'legacyFinetune', false);
if legacy
    bestVal = Inf;  warmEp = 1;                 % v6: no pretrain candidate, no warm-up
else
    bestVal = eval_val_loss(model, X, trainLabelsFull, valIdx, classIdxGlobal, lambda, hp.margin, chunk);
    % [v7] Gentler fine-tuning: lower lr, fresh Adam state, sep-loss warm-up.
    ahp.lr = ahp.lr * getf(hp, 'ftLrScale', 0.1);
    moments = init_adam_moments(model);
    t = 0;
    warmEp = getf(hp, 'sepWarmupEpochs', 20);
end

fprintf('[Stage2:finetune] start (maxEpochs=%d)\n', hp.maxEpochsFinetune);
for epoch = 1:hp.maxEpochsFinetune
    if epoch > 1 && mod(epoch, hp.pseudoT) == 0
        Hf = forward_hf(model, X, chunk);
        [P_idx, P_label] = generate_pseudo_labels(Hf, adjList, labeledIdx, labeledLabels, hp.pseudoRetain, K);
        classIdxGlobal = build_class_idx_cell(trainIdx, trainLabelsFull(trainIdx), P_idx, P_label, K);
    end

    perm = randperm(n);
    nB = min(ceil(n / bs), maxBpe);
    epochLoss = 0;
    for bi = 1:nB
        idx = perm((bi - 1) * bs + 1 : min(bi * bs, n));
        if nLab > 0 && ~isempty(trainIdx)
            extra = trainIdx(randperm(numel(trainIdx), min(nLab, numel(trainIdx))));
            idx = union(idx, extra);
            idx = idx(randperm(numel(idx)));
        end
        nb = numel(idx);
        Xb = X(:, idx);

        % local views of labels and class index sets
        labLocal   = trainLabOnly(idx);
        localTrain = find(labLocal > 0);
        map = zeros(1, n);           % always size n (fixes first-iter / growth bugs)
        map(idx) = 1:nb;
        sepLocal = cell(1, K);
        for k = 1:K
            loc = map(classIdxGlobal{k});
            sepLocal{k} = loc(loc > 0);
        end

        t = t + 1;
        lamE = lambda; lamE.sep = lambda.sep * min(1, epoch / max(warmEp, 1));
        out = psae_forward(model, Xb);
        l = compute_losses(out, Xb, labLocal, localTrain, sepLocal, lamE, hp.margin);
        g = psae_backward(model, out, Xb, labLocal, localTrain, sepLocal, lamE, hp.margin);
        [model, moments] = apply_adam_to_model(model, g, moments, t, ahp);
        epochLoss = epochLoss + l.total;
    end
    history.finetuneLoss(end + 1) = epochLoss / nB;

    % ---- validation on the full graph (chunked) ----
    valTotal = eval_val_loss(model, X, trainLabelsFull, valIdx, classIdxGlobal, lambda, hp.margin, chunk);
    history.finetuneValLoss(end + 1) = valTotal;
    if mod(epoch, 25) == 0
        fprintf('[Stage2:finetune] epoch %d/%d val=%.5f (%.0fs)\n', epoch, hp.maxEpochsFinetune, valTotal, toc(tStage2));
    end

    if valTotal < bestVal - 1e-6
        bestVal = valTotal; noImprove = 0; bestModel = model;
    else
        noImprove = noImprove + 1;
    end
    minFt = getf(hp, 'minEpochsFinetune', 20);
    if isempty(minFt), minFt = 20; end
    if noImprove >= hp.patience && epoch >= minFt
        fprintf('[Stage2-mb:finetune] early stop at epoch %d (val total=%.5f)\n', epoch, valTotal);
        break
    end
end

fprintf('[Stage2:finetune] done (epochs=%d, val=%.5f)\n', numel(history.finetuneValLoss), bestVal);
model = bestModel;
history.Hf = forward_hf(model, X, chunk);   % dF x n
end

% ======================================================================
function Hf = forward_hf(model, X, chunk)
n = size(X, 2);
Hf = [];
for s = 1:chunk:n
    e = min(s + chunk - 1, n);
    out = psae_forward(model, X(:, s:e));
    if isempty(Hf), Hf = zeros(size(out.Hf, 1), n); end
    Hf(:, s:e) = out.Hf;
end
end

function total = eval_val_loss(model, X, labelsFull, valIdx, classIdxGlobal, lambda, margin, chunk)
% Same quantity as compute_losses(...) on the full graph, but chunked:
% rec over all nodes, sup over valIdx, sep over the current class sets.
n = size(X, 2);
K = model.numClasses;
recSum = 0;
Hf = [];
for s = 1:chunk:n
    e = min(s + chunk - 1, n);
    out = psae_forward(model, X(:, s:e));
    d = out.Xhat - X(:, s:e);
    recSum = recSum + sum(d(:) .^ 2);
    if isempty(Hf), Hf = zeros(size(out.Hf, 1), n); end
    Hf(:, s:e) = out.Hf;
end
Lrec = recSum / n;

Lsup = 0;
if ~isempty(valIdx)
    logits = model.clf.W * Hf(:, valIdx) + model.clf.b;
    logits = logits - max(logits, [], 1);
    p = exp(logits); p = p ./ sum(p, 1);
    onehot = zeros(K, numel(valIdx));
    for j = 1:numel(valIdx)
        lab = labelsFull(valIdx(j));
        if lab >= 1 && lab <= K
            onehot(lab, j) = 1;
        end
    end
    Lsup = -sum(sum(onehot .* log(max(p, 1e-10)))) / numel(valIdx);
end
Lsep = separation_loss_and_grad(Hf, classIdxGlobal, margin);
total = lambda.rec * Lrec + lambda.sup * Lsup + lambda.sep * Lsep;
end

function s = adam_hp(hp)
s = struct( ...
    'lr',          getf(hp, 'lr', 1e-3), ...
    'beta1',       getf(hp, 'beta1', 0.9), ...
    'beta2',       getf(hp, 'beta2', 0.999), ...
    'eps',         getf(hp, 'epsAdam', 1e-8), ...
    'weightDecay', getf(hp, 'weightDecay', 1e-4));
end

function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
