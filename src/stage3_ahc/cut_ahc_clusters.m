function labels = cut_ahc_clusters(repMerges, n, K)
% CUT_AHC_CLUSTERS  Replay the first (n-K) merges from AHC's
%                   repMerges (sorted ascending by distance) to assign
%                   each of the n nodes a cluster label in {1..K}. This
%                   implements Algorithm 4's stopping rule
%                   ("while number of communities > K").
%
%   labels = CUT_AHC_CLUSTERS(repMerges, n, K)
%
%   If K is below the number of connected components of the candidate
%   graph (i.e. below the minimum reachable cluster count, since
%   disconnected nodes can never be merged by construction), K is
%   clipped up to that minimum and a warning is issued.

numMerges = size(repMerges, 1);
minK = n - numMerges;
if K < minK
    warning('cut_ahc_clusters:clippedK', ...
        ['Requested K=%d is below the minimum reachable %d (the candidate ' ...
         'graph has %d disconnected components); clipping to %d.'], ...
        K, minK, minK, minK);
    K = minK;
end

mergesToApply = min(n - K, numMerges);

parent = 1:n;
for m = 1:mergesToApply
    [ra, parent] = find_root(parent, repMerges(m, 1));
    [rb, parent] = find_root(parent, repMerges(m, 2));
    if ra ~= rb
        parent(ra) = rb;
    end
end

roots = zeros(1, n);
for xi = 1:n
    [roots(xi), parent] = find_root(parent, xi);
end
[~, ~, labels] = unique(roots);
labels = labels(:)';
end
