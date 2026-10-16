function A = adjlist_to_matrix(adjList)
% ADJLIST_TO_MATRIX  Build the dense n x n binary adjacency matrix from
%                     an adjacency-list cell array. Used as the raw
%                     structural input representation for the
%                     "-Node2Vec" ablation variant (Table
%                     "Average performance of PSAE-SSL and its ablation
%                     variants"): each node is represented by its raw
%                     connectivity row instead of a learned Node2Vec
%                     embedding, isolating the contribution of Stage 1.
%
%   A = ADJLIST_TO_MATRIX(adjList)

n = numel(adjList);
A = zeros(n, n);
for v = 1:n
    A(v, adjList{v}) = 1;
end
end
