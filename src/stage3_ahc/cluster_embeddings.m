function labels = cluster_embeddings(Hf, K, seed, adjList, varargin)
% CLUSTER_EMBEDDINGS  Paper-aligned Stage-3 clustering.
%   Uses only adjacency-constrained single-linkage AHC followed by a
%   modularity-optimal dendrogram cut (and optional isolate reassignment).
%   Spectral / k-means candidate generation and best-of-candidates selection
%   have been removed so that the final partition is produced exclusively
%   by the AHC procedure described in the paper.
%
%   When a target K is supplied the dendrogram is cut at that K.
%   When K is empty / omitted the cut is chosen by maximising modularity
%   over a plausible range (same logic as select_k_by_modularity).

if nargin < 3, seed = 1; end
if nargin < 4, adjList = []; end
rng(seed);

if size(Hf, 1) > size(Hf, 2) && size(Hf, 2) < 512
    Hf = Hf';
end
n = size(Hf, 2);
if nargin < 2 || isempty(K)
    K = [];
else
    K = max(1, min(K, n));
end

colN = sqrt(sum(Hf.^2, 1)) + 1e-12;
Hf = Hf ./ colN;

if isempty(adjList) || numel(adjList) ~= n
    % Fallback only when no graph is available
    if isempty(K), K = 2; end
    labels = simple_kmeans(Hf, K, seed);
    labels = labels(:)';
    return
end

% Build single-linkage dendrogram (adjacency-constrained)
[~, repMerges, mergeInfo] = ahc(Hf, adjList, true);

if isempty(K)
    % Paper default: choose K by maximising modularity
    minK = max(1, n - size(repMerges, 1));
    Krange = minK:min(n, max(minK + 200, 300));
    [labels, ~] = select_k_by_modularity(repMerges, n, adjList, Krange);
else
    labels = cut_ahc_clusters(repMerges, n, K);
end

labels = reassign_isolates(Hf, labels, max(labels));
labels = labels(:)';
end
