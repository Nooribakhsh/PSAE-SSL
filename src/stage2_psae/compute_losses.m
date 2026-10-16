function losses = compute_losses(out, X, labels, labeledIdx, classIdxCellForSep, lambda, margin)
% COMPUTE_LOSSES  Evaluate the PSAE training objective.

N = out.N;

% ---- Reconstruction loss ----
diffX = out.Xhat - X;
Lrec = sum(diffX(:) .^ 2) / max(N, 1);

% ---- Supervised cross-entropy, labeled nodes only ----
if isempty(labeledIdx)
    Lsup = 0;
else
    K = size(out.probs, 1);
    onehot = zeros(K, numel(labeledIdx));
    for j = 1:numel(labeledIdx)
        idx = labeledIdx(j);
        if idx < 1 || idx > numel(labels)
            continue
        end
        lab = labels(idx);
        if lab >= 1 && lab <= K
            onehot(lab, j) = 1;
        end
    end
    p = out.probs(:, labeledIdx);
    Lsup = -sum(sum(onehot .* log(max(p, 1e-10)))) / max(numel(labeledIdx), 1);
end

% ---- Separation loss ----
Lsep = separation_loss_and_grad(out.Hf, classIdxCellForSep, margin);

% ---- Total ----
Ltotal = lambda.rec * Lrec + lambda.sup * Lsup + lambda.sep * Lsep;

losses = struct('rec', Lrec, 'sup', Lsup, 'sep', Lsep, 'total', Ltotal);
end
