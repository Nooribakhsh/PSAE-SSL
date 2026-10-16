function adjList = load_graph(edgeFile, n, directed)
% LOAD_GRAPH  Read a two-column edge list and build an adjacency-list
%             representation G(V,E) as used throughout Section "Proposed
%             method" of the PSAE-SSL paper.
%
%   adjList = LOAD_GRAPH(edgeFile, n, directed)
%
%   Inputs
%     edgeFile : path to a whitespace-separated text file, one edge per
%                line: "src dst" (1-indexed node ids)
%     n        : number of nodes |V| in the graph (see Table "Overview of
%                benchmark datasets")
%     directed : logical, false (default) for undirected graphs
%                (Karate, Dolphin, Football, Polbooks, Cora, PubMed, DBLP
%                are undirected; Polblogs, Email, CiteSeer are directed
%                per the paper's dataset table)
%
%   Output
%     adjList : n x 1 cell array; adjList{v} is a row vector of the
%               (unique, sorted) neighbor ids of node v. Self-loops are
%               ignored.

if nargin < 3
    directed = false;
end

% Use textscan to avoid depending on readmatrix (removed polyfill that
% would shadow MATLAB R2023b's builtin — C3).
fid = fopen(edgeFile, 'r');
if fid < 0
    error('load_graph:openFailed', 'Cannot open edge file: %s', edgeFile);
end
C = textscan(fid, '%f %f', 'CommentStyle', '#', 'CollectOutput', true);
fclose(fid);
E = C{1};
if isempty(E) || size(E, 2) ~= 2
    error('load_graph:badFormat', ...
        'Edge file must have exactly two columns (src dst).');
end

adjList = cell(n, 1);
for k = 1:size(E, 1)
    a = E(k, 1);
    b = E(k, 2);
    if a == b
        continue % ignore self-loops
    end
    adjList{a} = [adjList{a}, b];
    if ~directed
        adjList{b} = [adjList{b}, a];
    end
end

for v = 1:n
    if isempty(adjList{v})
        adjList{v} = zeros(1, 0);  % consistent empty row
    else
        adjList{v} = unique(adjList{v}(:)');  % force row vector
    end
end
end
