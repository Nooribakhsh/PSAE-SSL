function [P_idx, P_label] = generate_pseudo_labels(Hf, adjList, labeledIdx, labeledLabels, retentionRate, K)
% GENERATE_PSEUDO_LABELS  Pseudo-label generation and refresh (Algorithm 4).
%
%   Aligns AHC cluster ids to ground-truth class ids via optimal assignment
%   on the contingency table of labeled nodes. Clusters with no labeled
%   nodes receive remaining class ids.

n = size(Hf, 2);

[~, repMerges, ~] = ahc(Hf, adjList, true);
clusterLabels = cut_ahc_clusters(repMerges, n, K);
clusterLabels = reassign_isolates(Hf, clusterLabels, K);

% contingency(c, k) = # labeled nodes in cluster c with true label k
cont = zeros(K, K);
for i = 1:numel(labeledIdx)
    c = clusterLabels(labeledIdx(i));
    k = labeledLabels(i);
    if c >= 1 && c <= K && k >= 1 && k <= K
        cont(c, k) = cont(c, k) + 1;
    end
end

% Maximize agreement: minimize cost = -cont
cost = -cont;
assignment = assign_clusters_to_classes(cost);

% Fill empty clusters (no labeled nodes) with remaining class ids
used = false(1, K);
for c = 1:K
    if assignment(c) >= 1 && assignment(c) <= K
        used(assignment(c)) = true;
    end
end
remain = find(~used);
ri = 1;
for c = 1:K
    if sum(cont(c, :)) == 0
        if ri <= numel(remain)
            assignment(c) = remain(ri);
            ri = ri + 1;
        else
            assignment(c) = c;
        end
    end
end

alignedLabels = zeros(1, n);
for c = 1:K
    alignedLabels(clusterLabels == c) = assignment(c);
end

P_idx = [];
P_label = [];
for c = 1:K
    members = find(alignedLabels == c);
    if isempty(members)
        continue
    end
    mu = mean(Hf(:, members), 2);
    d = sqrt(sum((Hf(:, members) - mu) .^ 2, 1));
    [~, order] = sort(d, 'ascend');
    numKeep = max(1, round(retentionRate * numel(members)));
    keep = members(order(1:numKeep));
    keep = setdiff(keep(:), labeledIdx(:));  % column
    keep = keep(:)';                         % force row for horzcat
    if ~isempty(keep)
        P_idx = [P_idx, keep]; %#ok<AGROW>
        P_label = [P_label, repmat(c, 1, numel(keep))]; %#ok<AGROW>
    end
end
end

function assignment = assign_clusters_to_classes(costMatrix)
% Optimal assignment: prefer matchpairs (Munkres), fall back to greedy.
n = size(costMatrix, 1);
assignment = zeros(1, n);
try
    % MATLAB Optimization Toolbox / R2019a+: matchpairs
    pairs = matchpairs(costMatrix, 1e6);
    for t = 1:size(pairs, 1)
        assignment(pairs(t, 1)) = pairs(t, 2);
    end
catch
    % Greedy fallback on reduced costs
    C = costMatrix;
    for i = 1:n
        C(i, :) = C(i, :) - min(C(i, :));
    end
    for j = 1:n
        C(:, j) = C(:, j) - min(C(:, j));
    end
    colUsed = false(1, n);
    [rows, cols] = find(abs(C) < 1e-12);
    scores = zeros(numel(rows), 1);
    for t = 1:numel(rows)
        scores(t) = -costMatrix(rows(t), cols(t));
    end
    [~, ord] = sort(scores, 'descend');
    for t = ord'
        r = rows(t); c = cols(t);
        if assignment(r) == 0 && ~colUsed(c)
            assignment(r) = c;
            colUsed(c) = true;
        end
    end
    freeCols = find(~colUsed);
    fi = 1;
    for r = 1:n
        if assignment(r) == 0
            if fi <= numel(freeCols)
                assignment(r) = freeCols(fi);
                fi = fi + 1;
            else
                assignment(r) = r;
            end
        end
    end
end
% Ensure every row assigned
for r = 1:n
    if assignment(r) < 1 || assignment(r) > n
        assignment(r) = r;
    end
end
end
