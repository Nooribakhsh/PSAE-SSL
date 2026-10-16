function [Z, repMerges, mergeInfo] = ahc(Hf, adjList, quiet)
% AHC  Stage 3 hierarchical clustering (paper-aligned).
%   Always uses adjacency-constrained single-linkage on graph edges
%   (sparsified AHC as described in Section 3.3 / Algorithm 6).
%   Candidate pairs are restricted to observed edges; distances are
%   Euclidean distances in the embedding space between adjacent nodes.
%   Components that would remain disconnected are bridged by nearest-
%   neighbour links in embedding space so that a complete dendrogram
%   (n-1 merges) is obtained and any K >= 1 is reachable.
%
%   quiet (optional): true -> suppress log (used by pseudo-label path)

if nargin < 3, quiet = false; end
n = size(Hf, 2);
if size(Hf, 1) == n && size(Hf, 2) < size(Hf, 1)
    Hf = Hf';
    n = size(Hf, 2);
end

if ~quiet
    fprintf('[Stage3:AHC] adjacency-constrained single-linkage dendrogram (n=%d)\n', n);
end

edges = adjlist_to_edgelist(adjList);
if isempty(edges)
    Z = zeros(0, 3);
    repMerges = zeros(0, 2);
    mergeInfo.numComponents = n;
    mergeInfo.method = 'single-empty';
    return
end

% Paper Section 3.3: candidate pairs come from graph adjacency only. When the
% graph has multiple connected components, pure adjacency-constrained AHC
% cannot merge across components. We add a minimal set of virtual bridges
% from each non-giant component to its nearest giant-component node in the
% fused embedding space so the dendrogram is complete (K>=1 reachable).
% These bridges are not original graph edges; they are only merge candidates.
edges = [edges; bridge_components(Hf, edges, n)];
m = size(edges, 1);
d = zeros(m, 1);
chunk = 200000;
for s = 1:chunk:m
    e = min(s + chunk - 1, m);
    diffs = Hf(:, edges(s:e, 1)) - Hf(:, edges(s:e, 2));
    d(s:e) = sqrt(sum(diffs .^ 2, 1))';
end
[dSorted, ord] = sort(d, 'ascend');
edgesSorted = edges(ord, :);
parent = 1:n;
clusterId = 1:n;
Z = zeros(n - 1, 3);
repMerges = zeros(n - 1, 2);
zRow = 0;
for e = 1:size(edgesSorted, 1)
    a = edgesSorted(e, 1); b = edgesSorted(e, 2);
    [ra, parent] = find_root(parent, a);
    [rb, parent] = find_root(parent, b);
    if ra ~= rb
        zRow = zRow + 1;
        Z(zRow, :) = [clusterId(ra), clusterId(rb), dSorted(e)];
        repMerges(zRow, :) = [ra, rb];
        parent(ra) = rb;
        clusterId(rb) = n + zRow;
    end
end
Z = Z(1:zRow, :);
repMerges = repMerges(1:zRow, :);
roots = zeros(1, n);
for xi = 1:n
    [roots(xi), parent] = find_root(parent, xi);
end
mergeInfo.numComponents = numel(unique(roots));
mergeInfo.method = 'single-adjacency';
end

function extra = bridge_components(Hf, edges, n)
% Extra (virtual) edges linking small components to the giant component.
extra = zeros(0, 2);
comp = component_labels(edges, n);
counts = accumarray(comp(:), 1);
if numel(counts) <= 1, return; end
[~, g] = max(counts);
outside = find(comp ~= g);
if numel(outside) > 5000
    warning('ahc:tooManyComponents', ...
        '%d nodes outside the giant component; bridging skipped.', numel(outside));
    return
end
giantIdx = find(comp == g);
Hg = Hf(:, giantIdx);
extra = zeros(numel(outside), 2);
for t = 1:numel(outside)
    v = outside(t);
    dv = sum((Hg - Hf(:, v)) .^ 2, 1);
    [~, j] = min(dv);
    u = giantIdx(j);
    extra(t, :) = [min(u, v), max(u, v)];
end
end

function comp = component_labels(edges, n)
% Connected-component id of every node (union-find, no toolbox needed).
parent = 1:n;
for e = 1:size(edges, 1)
    [ra, parent] = find_root(parent, edges(e, 1));
    [rb, parent] = find_root(parent, edges(e, 2));
    if ra ~= rb
        parent(ra) = rb;
    end
end
comp = zeros(1, n);
for i = 1:n
    [comp(i), parent] = find_root(parent, i);
end
[~, ~, comp] = unique(comp);
comp = comp(:)';
end
