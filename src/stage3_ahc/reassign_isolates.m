function labels = reassign_isolates(Hf, labels, Ktarget)
% REASSIGN_ISOLATES  When single-linkage AHC cannot merge disconnected
%   components, cut_ahc_clusters may return more than Ktarget clusters
%   (S2: Polblogs, CiteSeer, Cora). Map surplus singleton / tiny clusters
%   onto the Ktarget largest clusters by nearest centroid in Hf.
%
%   labels = REASSIGN_ISOLATES(Hf, labels, Ktarget)

labels = labels(:)';
n = numel(labels);
u = unique(labels);
if numel(u) <= Ktarget
    return
end

% Cluster sizes
sizes = zeros(size(u));
for i = 1:numel(u)
    sizes(i) = sum(labels == u(i));
end
[~, ord] = sort(sizes, 'descend');
keep = u(ord(1:Ktarget));           % retained community ids
drop = u(ord(Ktarget+1:end));       % to be absorbed

% Centroids of kept clusters
dF = size(Hf, 1);
mu = zeros(dF, Ktarget);
for j = 1:Ktarget
    idx = find(labels == keep(j));
    mu(:, j) = mean(Hf(:, idx), 2);
end

% Map drop members to nearest kept centroid
for i = 1:numel(drop)
    idx = find(labels == drop(i));
    for t = 1:numel(idx)
        v = Hf(:, idx(t));
        dist = sum((mu - v) .^ 2, 1);
        [~, j] = min(dist);
        labels(idx(t)) = keep(j);
    end
end

% Relabel to contiguous 1..Ktarget
[~, ~, labels] = unique(labels);
labels = labels(:)';
end
