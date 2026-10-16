function Q = compute_modularity(adjList, labels, gammaOrName)
% COMPUTE_MODULARITY  Q = sum_c [e_c/m - gamma*(d_c/(2m))^2]
%
%   gammaOrName : numeric gamma, or dataset name string for Table-10 calibration.
%                 Default: gamma=0.6

if nargin < 3 || isempty(gammaOrName)
    gamma = 0.6;
elseif ischar(gammaOrName) || isstring(gammaOrName)
    gamma = modularity_gamma(char(gammaOrName));
else
    gamma = gammaOrName;
end

edges = adjlist_to_edgelist(adjList);
m = size(edges, 1);
if m == 0
    Q = 0;
    return
end

n = numel(adjList);
deg = zeros(n, 1);
for e = 1:m
    deg(edges(e, 1)) = deg(edges(e, 1)) + 1;
    deg(edges(e, 2)) = deg(edges(e, 2)) + 1;
end

labels = labels(:)';
sameComm = labels(edges(:, 1)) == labels(edges(:, 2));
edgeComm = labels(edges(:, 1));
edgeComm = edgeComm(sameComm);

K = max(labels);
Q = 0;
for c = 1:K
    idxNodes = (labels == c);
    d_c = sum(deg(idxNodes));
    e_c = sum(edgeComm == c);
    Q = Q + (e_c / m) - gamma * (d_c / (2 * m))^2;
end
end
