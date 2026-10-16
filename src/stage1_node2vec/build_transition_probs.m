function G = build_transition_probs(adjList, p, q)
% BUILD_TRANSITION_PROBS  Precompute the biased 2nd-order random-walk
%                          transition tables used by Node2Vec (paper
%                          Section "Stage 1: Node embedding using
%                          Node2Vec", Algorithm 1).
%
%   G = BUILD_TRANSITION_PROBS(adjList, p, q)
%
%   Inputs
%     adjList : n x 1 cell array from LOAD_GRAPH (unweighted graph:
%               all edge weights w_vx = 1)
%     p       : return parameter (Equation eq_weight) — penalizes
%               immediate backtracking when small
%     q       : in-out parameter (Equation eq_weight) — q<1 -> DFS-like
%               (global) exploration, q>1 -> BFS-like (local) exploration
%
%   Output
%     G : struct with fields
%           .adjList     - the original adjacency list
%           .alias_nodes - n x 1 cell of struct('J','q','neighbors')
%                          used to sample the FIRST step of a walk
%                          (uniform for an unweighted graph)
%           .edgeId / .alias_edge_list - sparse edge-id matrix and a cell
%                          array of struct('J','q','neighbors'); the table
%                          of directed edge (t,v) (previous node t, current
%                          node v) is alias_edge_list{edgeId(t,v)} and implements the biased
%                          transition probability p(c_i=x | c_{i-1}=v)
%                          of Equation eq_prob, with unnormalized weight
%                          alpha_{p,q}(t,x) from Equation eq_weight:
%                             1/p  if d(t,x) = 0  (x == t)
%                             1    if d(t,x) = 1  (x is a neighbor of t)
%                             1/q  if d(t,x) = 2  (otherwise)
%
%   Note on scalability: for very large graphs (e.g. DBLP, |V| ~ 317K)
%   replace the ismember() neighbor check below with a logical/sparse
%   membership vector per node to keep edge-table construction close to
%   linear in the number of edges, consistent with the paper's stated
%   O(n*r*l) complexity for Stage 1.

n = numel(adjList);

% ---- 1st-order alias tables (uniform start distribution) ----
G.alias_nodes = cell(n, 1);
for v = 1:n
    nbrs = adjList{v};
    if isempty(nbrs)
        continue
    end
    probs = ones(1, numel(nbrs)) / numel(nbrs);
    [J, qq] = alias_setup(probs);
    G.alias_nodes{v} = struct('J', J, 'q', qq, 'neighbors', nbrs);
end

% ---- 2nd-order alias tables, one per directed edge (t -> v) ----
% Same transition probabilities as before (1/p if x==t, 1 if x is a
% neighbor of t, 1/q otherwise). For speed, tables are stored in a cell
% array indexed through a sparse edge-id matrix (no string keys), and the
% "x is a neighbor of t" test uses a logical mark vector.
edgeT = zeros(0, 1); edgeV = zeros(0, 1);
edgeTab = {};
mark = false(n, 1);
nEdge = 0;
for t = 1:n
    nbrs_t = adjList{t};
    if isempty(nbrs_t)
        continue
    end
    nbrs_t = nbrs_t(:)';   % force row
    valid = nbrs_t(nbrs_t >= 1 & nbrs_t <= n);
    mark(valid) = true;
    for vi = 1:numel(nbrs_t)
        v = nbrs_t(vi);
        if v < 1 || v > n
            continue
        end
        nbrs_v = adjList{v};
        if isempty(nbrs_v)
            continue
        end
        nbrs_v = nbrs_v(:)';
        unnorm = (1 / q) * ones(1, numel(nbrs_v));      % d_tx = 2
        inRange = nbrs_v >= 1 & nbrs_v <= n;
        isNbr = false(1, numel(nbrs_v));
        isNbr(inRange) = mark(nbrs_v(inRange))';
        unnorm(isNbr) = 1;                              % d_tx = 1
        unnorm(nbrs_v == t) = 1 / p;                    % d_tx = 0
        s = sum(unnorm);
        if s <= 0
            probs = ones(1, numel(nbrs_v)) / numel(nbrs_v);
        else
            probs = unnorm / s;
        end
        [J, qq] = alias_setup(probs);
        if isempty(J)
            continue
        end
        nEdge = nEdge + 1;
        edgeT(nEdge, 1) = t; %#ok<AGROW>
        edgeV(nEdge, 1) = v; %#ok<AGROW>
        edgeTab{nEdge, 1} = struct('J', J, 'q', qq, 'neighbors', nbrs_v); %#ok<AGROW>
    end
    mark(valid) = false;
end

% edgeId(t, v) = index into alias_edge_list (0 if no table for that edge).
% sparse() would sum duplicate (t,v) pairs, so keep the last id per pair.
[pairKey, ia] = unique(edgeT * (n + 1) + edgeV, 'last');
G.edgeId = sparse(edgeT(ia), edgeV(ia), ia, n, n);
G.alias_edge_list = edgeTab;
clear pairKey

G.adjList = adjList;
end
