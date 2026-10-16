function edges = adjlist_to_edgelist(adjList)
% ADJLIST_TO_EDGELIST  Unique undirected edges [min max], self-loops removed,
%   sorted by (min, max). Same output (same order) as the previous loop-based
%   version, but vectorized and sparse (fast and memory-light for m ~ 1e6).
A = adjlist_to_sparse(adjList);
A = triu(max(A, A'), 1);
[r, c] = find(A);
edges = sortrows([r, c]);
end
