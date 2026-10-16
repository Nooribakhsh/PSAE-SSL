function grads = psae_backward(model, out, X, labels, labeledIdx, classIdxCellForSep, lambda, margin)
% PSAE_BACKWARD  Backprop the total loss (Equation eq_tot) through the
%                classifier, fusion layer, and every branch's
%                decoder/encoder.
%
% During unsupervised pretraining, call with lambda.sup = lambda.sep = 0
% and labeledIdx = [] — the classifier/separation gradients vanish and
% this reduces exactly to independent per-branch autoencoder backprop,
% i.e. Algorithm 3's pretraining step.
%
%   grads : struct with fields branches (1 x B cell of per-branch
%           grads, see BRANCH_BACKWARD), fusion, clf — mirrors the
%           shape of `model` for APPLY_ADAM_TO_MODEL.

N = out.N;
B = model.B;

% ---- dL/dXhat from L_rec (Equation eq_rec) ----
dXhat = lambda.rec * (2 / N) * (out.Xhat - X);
dXhat_b_common = dXhat / B; % Xhat = (1/B) sum_b Xhat_b

% ---- dL/dHf from L_sup (classifier), Equation eq_sup ----
K = model.numClasses;
dLogits = zeros(K, N);
if ~isempty(labeledIdx)
    onehot = zeros(K, numel(labeledIdx));
    for j = 1:numel(labeledIdx)
        onehot(labels(labeledIdx(j)), j) = 1;
    end
    p = out.probs(:, labeledIdx);
    dLogits(:, labeledIdx) = lambda.sup * (p - onehot) / numel(labeledIdx);
end
dHf_sup = model.clf.W' * dLogits;
dW_clf  = dLogits * out.Hf';
db_clf  = sum(dLogits, 2);

% ---- dL/dHf from L_sep, Equation eq_sep ----
[~, dHf_sep_raw] = separation_loss_and_grad(out.Hf, classIdxCellForSep, margin);
dHf_sep = lambda.sep * dHf_sep_raw;

dHf = dHf_sup + dHf_sep;

% ---- backprop through fusion layer ----
[dW_fusion, db_fusion, dHcat] = dense_backward(dHf, out.fusionCache, model.fusion.W);

% ---- split dHcat back into per-branch latent gradients, then backprop each branch ----
grads.branches = cell(B, 1);
rowStart = 1;
for b = 1:B
    latentDim = size(out.Hb_all{b}, 1);
    dHb_extra = dHcat(rowStart:rowStart + latentDim - 1, :);
    rowStart = rowStart + latentDim;

    grads.branches{b} = branch_backward(model.branches{b}, out.branchCaches{b}, dXhat_b_common, dHb_extra);
end

grads.fusion = struct('dW', dW_fusion, 'db', db_fusion);
grads.clf    = struct('dW', dW_clf,    'db', db_clf);
end
