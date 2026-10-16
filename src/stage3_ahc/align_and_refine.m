function labels = align_and_refine(Hf, labels, labeledIdx, labeledLabels, K, adjList)
% ALIGN_AND_REFINE  Map cluster ids to GT on L, force L, reassign U by centroids.
labels = labels(:)';
n = numel(labels);
if size(Hf, 2) ~= n && size(Hf, 1) == n
    Hf = Hf';
end

% Contingency pred->true on labeled
cont = zeros(K, K);
for i = 1:numel(labeledIdx)
    p = labels(labeledIdx(i));
    t = labeledLabels(i);
    if p >= 1 && p <= K && t >= 1 && t <= K
        cont(p, t) = cont(p, t) + 1;
    end
end
usedT = false(1, K); usedP = false(1, K);
map = zeros(1, K);
for iter = 1:K
    best = -1; bp = 1; bt = 1;
    for p = 1:K
        if usedP(p), continue; end
        for tt = 1:K
            if usedT(tt), continue; end
            if cont(p, tt) > best
                best = cont(p, tt); bp = p; bt = tt;
            end
        end
    end
    map(bp) = bt; usedP(bp) = true; usedT(bt) = true;
end
for p = 1:K
    if map(p) == 0, map(p) = p; end
end
newL = labels;
for i = 1:n
    if labels(i) >= 1 && labels(i) <= K
        newL(i) = map(labels(i));
    end
end
labels = newL;

% Force + centroid reassignment
labels = refine_with_labels(Hf, labels, labeledIdx, labeledLabels, K);

% Optional: one pass of modularity-preserving neighbor majority among unlabeled
if nargin >= 6 && ~isempty(adjList)
    labels = neighbor_majority_unlabeled(adjList, labels, labeledIdx, K);
end
labels = labels(:)';
end

function labels = neighbor_majority_unlabeled(adjList, labels, labeledIdx, K)
n = numel(labels);
isLab = false(1, n);
isLab(labeledIdx) = true;
newL = labels;
for v = 1:n
    if isLab(v), continue; end
    nbr = adjList{v};
    if isempty(nbr), continue; end
    counts = zeros(1, K);
    for u = nbr
        if u >= 1 && u <= n && labels(u) >= 1 && labels(u) <= K
            counts(labels(u)) = counts(labels(u)) + 1;
        end
    end
    if any(counts)
        [~, c] = max(counts);
        newL(v) = c;
    end
end
% Keep labeled fixed
for i = 1:numel(labeledIdx)
    newL(labeledIdx(i)) = labels(labeledIdx(i));
end
labels = newL;
end
