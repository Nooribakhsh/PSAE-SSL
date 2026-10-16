function score = node_cluster_affinity(Hf, i, members, epsAssign)
% NODE_CLUSTER_AFFINITY  Mean inverse-distance affinity of node i to cluster.
%   Using the mean (not the sum) avoids a systematic bias toward large clusters.

members = members(members ~= i);
if isempty(members)
    score = 0;
    return
end
d = sqrt(sum((Hf(:, i) - Hf(:, members)) .^ 2, 1));
score = mean(1 ./ (d + epsAssign));
end
