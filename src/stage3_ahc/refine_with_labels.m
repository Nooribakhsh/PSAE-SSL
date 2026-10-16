function labels = refine_with_labels(Hf, labels, labeledIdx, labeledLabels, K)
% REFINE_WITH_LABELS  Semi-supervised post-process:
%   1) Force ground-truth labels on L
%   2) Reassign unlabeled nodes by nearest labeled centroid in Hf

labels = labels(:)';
n = size(Hf, 2);
if size(Hf, 1) == n && size(Hf, 2) < size(Hf, 1)
    Hf = Hf';
end
if isempty(labeledIdx)
    return
end

% Align embedding orientation
if size(Hf, 2) ~= n
    if size(Hf, 1) == n
        Hf = Hf';
    else
        return
    end
end

% Force labeled nodes
for i = 1:numel(labeledIdx)
    v = labeledIdx(i);
    if v >= 1 && v <= numel(labels)
        labels(v) = labeledLabels(i);
    end
end

% Centroids from labeled nodes per class
cents = zeros(size(Hf, 1), K);
counts = zeros(1, K);
for i = 1:numel(labeledIdx)
    v = labeledIdx(i);
    c = labeledLabels(i);
    if c >= 1 && c <= K
        cents(:, c) = cents(:, c) + Hf(:, v);
        counts(c) = counts(c) + 1;
    end
end
for c = 1:K
    if counts(c) > 0
        cents(:, c) = cents(:, c) / counts(c);
    else
        % fallback: mean of current cluster
        idx = find(labels == c);
        if ~isempty(idx)
            cents(:, c) = mean(Hf(:, idx), 2);
        end
    end
end

% Unlabeled nodes -> nearest centroid
isLab = false(1, n);
isLab(labeledIdx) = true;
for v = 1:n
    if isLab(v), continue; end
    best = labels(v);
    bestD = Inf;
    for c = 1:K
        d = norm(Hf(:, v) - cents(:, c));
        if d < bestD
            bestD = d;
            best = c;
        end
    end
    labels(v) = best;
end
labels = labels(:)';
end
