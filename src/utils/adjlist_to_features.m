function Z0 = adjlist_to_features(adjList, targetDim, seed)
% ADJLIST_TO_FEATURES  Sparse structural features for the -Node2Vec ablation.
%
%   Z0 = ADJLIST_TO_FEATURES(adjList, targetDim, seed)
%
%   Builds a sparse adjacency, then a fixed-dimensional node feature matrix
%   (n x targetDim) suitable as PSAE input without materializing a dense
%   n x n matrix (E4):
%     1) degree (log1p) and clustering-proxy (shared-neighbor count via A*A
%        diagonal estimate is skipped for speed; we use degree only)
%     2) Johnson-Lindenstrauss-style random projection: A_sym * R
%        where R is n x (targetDim-1) Gaussian, so neighborhood structure
%        is approximately preserved in O(m * d) time and memory.
%
%   For small n (<= targetDim) returns the dense adjacency rows (exact).

if nargin < 2 || isempty(targetDim), targetDim = 128; end
if nargin < 3 || isempty(seed), seed = 1; end

n = numel(adjList);
A = adjlist_to_sparse(adjList);          % n x n sparse, directed as stored
A = spones(max(A, A'));                  % undirected binary for features

if n <= targetDim
    Z0 = full(A);                        % exact rows when cheap
    return
end

rng(seed);
dProj = max(targetDim - 1, 1);
% Gaussian JL projection: (A * R) / sqrt(dProj)
R = randn(n, dProj) / sqrt(dProj);
proj = A * R;                            % sparse * dense -> n x dProj
deg = full(sum(A, 2));
logDeg = log1p(deg);
Z0 = [logDeg, full(proj)];               % n x targetDim
% Column-wise z-score for scale stability
mu = mean(Z0, 1);
sd = std(Z0, 0, 1);
sd(sd < 1e-8) = 1;
Z0 = (Z0 - mu) ./ sd;
end
