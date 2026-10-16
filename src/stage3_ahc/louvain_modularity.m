function [labels, Qbest] = louvain_modularity(adjList, seed)
% LOUVAIN_MODULARITY  Greedy modularity maximization (Newman Q, Eq. 17).
% Two-phase Louvain-style: local moves then aggregate. Multiple random
% node orders. Returns the highest-Q partition found.

if nargin < 2, seed = 1; end
rng(seed);

n = numel(adjList);
edges = adjlist_to_edgelist(adjList);
m = size(edges, 1);
if m == 0
    labels = ones(1, n);
    Qbest = 0;
    return
end

% Build neighbor lists + undirected adjacency weight 1
nbrs = cell(n, 1);
for e = 1:m
    a = edges(e,1); b = edges(e,2);
    nbrs{a}(end+1) = b; %#ok<AGROW>
    nbrs{b}(end+1) = a; %#ok<AGROW>
end
deg = zeros(n, 1);
for i = 1:n
    nbrs{i} = unique(nbrs{i});
    deg(i) = numel(nbrs{i});
end
twoM = 2 * m;

bestLabels = ones(1, n);
Qbest = -Inf;

for restart = 0:4
    labels = 1:n;          % each node its own community
    improved = true;
    pass = 0;
    while improved && pass < 50
        improved = false;
        pass = pass + 1;
        order = randperm(n);
        for ii = 1:n
            i = order(ii);
            ci = labels(i);
            % community total degree
            % gain for moving i to neighbor communities
            neighCom = unique(labels(nbrs{i}));
            bestGain = 0;
            bestC = ci;
            % remove i from ci contribution
            for c = neighCom(:)'
                if c == ci, continue; end
                gain = mod_gain(i, ci, c, labels, nbrs, deg, twoM);
                if gain > bestGain + 1e-15
                    bestGain = gain;
                    bestC = c;
                end
            end
            if bestC ~= ci
                labels(i) = bestC;
                improved = true;
            end
        end
    end
    % renumber
    [~, ~, labels] = unique(labels);
    labels = labels(:)';
    Q = compute_modularity(adjList, labels);
    if Q > Qbest
        Qbest = Q;
        bestLabels = labels;
    end
end
labels = bestLabels;
end

function gain = mod_gain(i, cFrom, cTo, labels, nbrs, deg, twoM)
% Approximate delta-Q for moving node i from cFrom to cTo
ki = deg(i);
% links from i to cTo and cFrom
kTo = 0; kFrom = 0;
for v = nbrs{i}
    if labels(v) == cTo, kTo = kTo + 1; end
    if labels(v) == cFrom && v ~= i, kFrom = kFrom + 1; end
end
% community degrees (excluding i for from)
sumTo = 0; sumFrom = 0;
for u = 1:numel(labels)
    if labels(u) == cTo, sumTo = sumTo + deg(u); end
    if labels(u) == cFrom && u ~= i, sumFrom = sumFrom + deg(u); end
end
% delta Q * twoM (sign only needed; scale consistent)
% remove from cFrom: -kFrom/m + (sumFrom*ki)/m^2  roughly
% add to cTo: +kTo/m - (sumTo*ki)/m^2
gain = (kTo - kFrom) / twoM - (ki / twoM) * ((sumTo - sumFrom) / twoM);
end
