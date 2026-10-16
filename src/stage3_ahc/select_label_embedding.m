function [Hsel, info] = select_label_embedding(Hf, Z0, adjList, labeledIdx, labeledLabels, classList, baseLabels, seed)
% SELECT_LABEL_EMBEDDING  Choose the embedding used by the label-guided
% Stage-3 refinement, using ONLY the labeled nodes (hold-out accuracy).
%
%   Candidates (all d x n, unit columns):
%     1) Hf            fused PSAE embedding           (default)
%     2) [Hf; Z0n]     PSAE + Node2Vec concatenation
%     3) Z0n           Node2Vec only
%   A fused embedding that collapsed (all nodes ~ same vector) scores badly
%   on the held-out labeled nodes and is therefore not selected. No ground
%   truth of unlabeled nodes is used. Ties -> candidate 1.
%
%   Hf : d x n ; Z0 : n x dz (Stage-1 embedding) ; other args as in
%   LABEL_GUIDED_CLUSTERING.

n = size(Hf, 2);
Hsel = Hf;
info = struct('choice', 1, 'acc', nan(1, 3), 'collapsed', false);
info.collapsed = mean(std(Hf, 0, 2)) < 1e-3;

lab = zeros(1, numel(labeledIdx));
[~, lab] = ismember(labeledLabels(:)', classList(:)');
labeledIdx = labeledIdx(:)';
keep = lab > 0;  labeledIdx = labeledIdx(keep);  lab = lab(keep);
K = numel(classList);
if isempty(Z0) || size(Z0, 1) ~= n || numel(labeledIdx) < 2 * K
    return    % too few labels for a reliable hold-out: keep Hf
end

Zn = Z0' ./ (sqrt(sum(Z0'.^2, 1)) + 1e-12);
Hc = [Hf; Zn];  Hc = Hc ./ (sqrt(sum(Hc.^2, 1)) + 1e-12);
cands = {Hf, Hc, Zn};

% stratified hold-out of ~30% of the labeled nodes (>=1 train node / class)
rng(seed + 4242);
hold = false(1, numel(labeledIdx));
for c = 1:K
    mem = find(lab == c);
    nh = floor(0.3 * numel(mem));
    nh = min(nh, numel(mem) - 1);
    if nh >= 1, hold(mem(randperm(numel(mem), nh))) = true; end
end
if sum(hold) < 2, return; end

acc = nan(1, 3);
for ci = 1:3
    try
        pl = label_guided_clustering(cands{ci}, adjList, labeledIdx(~hold), lab(~hold), ...
            1:K, baseLabels, seed);
        acc(ci) = mean(pl(labeledIdx(hold)) == lab(hold));
    catch
        acc(ci) = -1;
    end
end
info.acc = acc;
[best, j] = max(acc);
if acc(1) >= best - 1e-12, j = 1; end     % tie -> default Hf
info.choice = j;
Hsel = cands{j};
end
