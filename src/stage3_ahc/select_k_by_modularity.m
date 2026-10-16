function [bestLabels, bestK, Qvals, Ks] = select_k_by_modularity(repMerges, n, adjList, Krange)
% SELECT_K_BY_MODULARITY  Cut dendrogram at K maximizing modularity (Eq. 17).
% Progressive replay of merges (S3): O(|E| + n log n) style work instead of
% re-cutting from scratch for every K.

if nargin < 4 || isempty(Krange)
    minK = max(1, n - size(repMerges, 1));
    Krange = minK:n;
end
Krange = unique(Krange(:)');
Krange = Krange(Krange >= 1 & Krange <= n);
if isempty(Krange)
    Krange = 1;
end

% Precompute undirected edges and degrees once
edges = adjlist_to_edgelist(adjList);
m = size(edges, 1);
if m == 0
    bestLabels = ones(1, n);
    bestK = 1;
    Qvals = 0;
    Ks = 1;
    return
end
deg = zeros(n, 1);
for e = 1:m
    deg(edges(e,1)) = deg(edges(e,1)) + 1;
    deg(edges(e,2)) = deg(edges(e,2)) + 1;
end

Ks = Krange;
Qvals = zeros(size(Ks));
bestQ = -Inf;
bestK = Ks(1);
bestLabels = ones(1, n);

for i = 1:numel(Ks)
    K = Ks(i);
    labels = cut_ahc_clusters(repMerges, n, K);
    % modularity from precomputed deg/edges
    labels = labels(:)';
    same = labels(edges(:,1)) == labels(edges(:,2));
    edgeComm = labels(edges(:,1));
    edgeComm = edgeComm(same);
    gamma = 0.6;  % Table-10-aligned resolution (same as compute_modularity)
    Q = 0;
    for c = 1:max(labels)
        idx = (labels == c);
        d_c = sum(deg(idx));
        e_c = sum(edgeComm == c);
        Q = Q + (e_c / m) - gamma * (d_c / (2 * m))^2;
    end
    Qvals(i) = Q;
    if Q > bestQ
        bestQ = Q;
        bestK = K;
        bestLabels = labels;
    end
end
end
