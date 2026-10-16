function A = adjlist_to_sparse(adjList)
% ADJLIST_TO_SPARSE  Sparse binary adjacency, same content as ADJLIST_TO_MATRIX
%   (A(v, adjList{v}) = 1, directed as stored, duplicates collapse to 1) but
%   as an n x n sparse logical-valued matrix. Memory ~ O(m) instead of O(n^2).
n = numel(adjList);
len = cellfun(@numel, adjList);
i = repelem((1:n)', len(:));
j = cell2mat(cellfun(@(x) x(:), adjList(:), 'UniformOutput', false));
A = spones(sparse(i, double(j), 1, n, n));
end
