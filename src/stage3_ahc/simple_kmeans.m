function labels = simple_kmeans(H, K, seed, maxIter)
% SIMPLE_KMEANS  Lloyd's k-means with random-sample initialization.
%   H : d x n (columns = points), K : #clusters.
%   Small problems (K*n <= 2e7): the original implementation, unchanged.
%   Large problems (e.g. DBLP: K = 13,477, n = 317K, where the K x n distance
%   matrix alone would take ~34 GB): same algorithm, same initialization and
%   same stopping rule, but distances are computed in column chunks
%   (argmin of ||c||^2 - 2 c'h, which equals the argmin of ||h-c||^2) and the
%   centroid update uses a sparse indicator matrix instead of a K-loop.
if nargin < 3 || isempty(seed), seed = 1; end
if nargin < 4 || isempty(maxIter), maxIter = 100; end
rng(seed);

n = size(H, 2);
initIdx = randperm(n, K);
centroids = H(:, initIdx);
labels = zeros(1, n);

if K * n <= 2e7
    % ---------- original dense implementation ----------
    for iter = 1:maxIter
        dists = zeros(K, n);
        for k = 1:K
            dists(k, :) = sum((H - centroids(:, k)) .^ 2, 1);
        end
        [~, newLabels] = min(dists, [], 1);
        if isequal(newLabels, labels), break; end
        labels = newLabels;
        for k = 1:K
            members = (labels == k);
            if any(members)
                centroids(:, k) = mean(H(:, members), 2);
            end
        end
    end
    return
end

% ---------- chunked implementation for large K*n ----------
chunk = max(1, floor(2e7 / K));
for iter = 1:maxIter
    cn = sum(centroids .^ 2, 1)';                   % K x 1
    newLabels = zeros(1, n);
    for s = 1:chunk:n
        e = min(s + chunk - 1, n);
        D = cn - 2 * (centroids' * H(:, s:e));      % K x chunk
        [~, newLabels(s:e)] = min(D, [], 1);
    end
    if isequal(newLabels, labels), break; end
    labels = newLabels;
    M = sparse(labels, 1:n, 1, K, n);               % K x n indicator
    counts = full(sum(M, 2))';                      % 1 x K
    nz = counts > 0;
    sums = H * M';                                  % d x K
    centroids(:, nz) = sums(:, nz) ./ counts(nz);   % empty clusters keep old centroid
end
end
