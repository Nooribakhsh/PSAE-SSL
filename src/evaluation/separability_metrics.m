function metrics = separability_metrics(H, labels)
% SEPARABILITY_METRICS  Geometric evaluation of a representation's
%                        community separability (Section "Latent space
%                        separability"), computing all four metrics
%                        reported in Table "Latent space separability
%                        before and after PSAE-SSL":
%
%   FDR (Equation eq_FDR) = D_inter / D_intra
%     D_inter (Equation eq_inter): mean pairwise centroid distance
%     D_intra (Equation eq_intra): mean node-to-own-centroid distance
%
%   SC  (Equation eq_SC): mean Silhouette Coefficient,
%     SC_i = (b(i) - a(i)) / max(a(i), b(i))
%
%   DBI (Equation eq_DBI): mean over communities of the worst-case
%     ratio of combined within-community dispersion to centroid
%     distance
%
%   CHI (Equation eq_CHI): ratio of between-community to
%     within-community variance (Calinski-Harabasz index)
%
%   metrics = SEPARABILITY_METRICS(H, labels)
%
%   H      : d x n embedding matrix (columns = nodes) — e.g. Z0' for the
%            Node2Vec representation, or Hf for the PSAE-SSL fused
%            representation
%   labels : 1 x n community assignment in {1..K}
%
%   metrics : struct with fields FDR, SC, DBI, CHI

[d, n] = size(H); %#ok<ASGLU>

% Reassignment steps upstream of this function (e.g. POST_CUT_REASSIGNMENT)
% can leave a cluster id with zero members. Remap to a dense 1..K
% range over only the labels actually present, so no empty cluster ever
% contributes a NaN centroid.
[~, ~, labels] = unique(labels(:)');
labels = labels(:)';
K = max(labels);

mu = zeros(d, K);
sigma = zeros(1, K);
for k = 1:K
    members = (labels == k);
    mu(:, k) = mean(H(:, members), 2);
    dists = sqrt(sum((H(:, members) - mu(:, k)) .^ 2, 1));
    sigma(k) = mean(dists);
end

%% D_inter, D_intra, FDR (Equations eq_inter, eq_intra, eq_FDR)
pairSum = 0;
pairCount = 0;
for a = 1:K - 1
    for b = a + 1:K
        pairSum = pairSum + norm(mu(:, a) - mu(:, b));
        pairCount = pairCount + 1;
    end
end
D_inter = pairSum / pairCount;
D_intra = mean(sigma);
metrics.FDR = D_inter / D_intra;

%% Silhouette Coefficient (Equation eq_SC)
s = zeros(1, n);
for i = 1:n
    ci = labels(i);
    sameMembers = find(labels == ci);
    sameMembers = sameMembers(sameMembers ~= i);
    if isempty(sameMembers)
        a_i = 0;
    else
        a_i = mean(sqrt(sum((H(:, i) - H(:, sameMembers)) .^ 2, 1)));
    end

    b_i = Inf;
    for k = 1:K
        if k == ci
            continue
        end
        otherMembers = find(labels == k);
        if isempty(otherMembers)
            continue
        end
        d_ik = mean(sqrt(sum((H(:, i) - H(:, otherMembers)) .^ 2, 1)));
        b_i = min(b_i, d_ik);
    end

    denom = max(a_i, b_i);
    if denom == 0
        s(i) = 0;
    else
        s(i) = (b_i - a_i) / denom;
    end
end
metrics.SC = mean(s);

%% Davies-Bouldin Index (Equation eq_DBI)
dbiTerms = zeros(1, K);
for i = 1:K
    best = -Inf;
    for j = 1:K
        if j == i
            continue
        end
        val = (sigma(i) + sigma(j)) / norm(mu(:, i) - mu(:, j));
        best = max(best, val);
    end
    dbiTerms(i) = best;
end
metrics.DBI = mean(dbiTerms);

%% Calinski-Harabasz Index (Equation eq_CHI)
grandMean = mean(H, 2);
SSB = 0;
for k = 1:K
    n_k = sum(labels == k);
    SSB = SSB + n_k * sum((mu(:, k) - grandMean) .^ 2);
end
SSW = 0;
for k = 1:K
    members = (labels == k);
    SSW = SSW + sum(sum((H(:, members) - mu(:, k)) .^ 2, 1));
end
metrics.CHI = (SSB / (K - 1)) / (SSW / (n - K));
end
